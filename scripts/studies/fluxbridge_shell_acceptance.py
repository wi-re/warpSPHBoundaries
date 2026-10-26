"""Near-roundoff 3D planar quadrature and matched CPU/CUDA analytic costs."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import warp as wp
from warpSPHBoundaries.fluxbridge import Request, baseline, monomials
from warpSPHBoundaries.fluxbridge.planar import PlanarEvaluator

ROOT = Path(__file__).resolve().parents[2]


def sources():
    paths = list((ROOT/"src/warpSPHBoundaries/fluxbridge").glob("*.py"))
    paths += list((ROOT/"tests/fluxbridge").glob("*.py"))
    paths += [Path(__file__), ROOT/"tests/fixtures/fluxbridge-planar-v1.json"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def timed(call, device, repeats=5, events=False):
    result = None; samples = []; device_samples = []
    for i in range(repeats):
        wp.synchronize_device(device)
        if events and device.is_cuda:
            start, end = wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)
            with wp.ScopedDevice(device): wp.record_event(start)
        t = perf_counter(); result = call()
        if events and device.is_cuda:
            with wp.ScopedDevice(device): wp.record_event(end)
        wp.synchronize_device(device)
        samples.append((perf_counter()-t)*1e3)
        if events and device.is_cuda: device_samples.append(wp.get_event_elapsed_time(start, end))
    return result, dict(median_ms=float(np.median(samples)), samples_ms=samples,
        cuda_event_median_ms=float(np.median(device_samples)) if device_samples else None)


def accuracy(device):
    cases = json.loads((ROOT/"tests/fixtures/fluxbridge-planar-v1.json").read_text())["cases"]
    rows = []
    for precision, budget in (("float64", 1e-14), ("float32", 1e-5)):
        for kernel in ("w2", "w4", "w6", "cubic"):
            errors = []; convergence = []; count = 0
            for c in cases:
                if c["dimension"] != 3 or c["kernel"] != kernel: continue
                count += 1; values = []
                for nodes in (16, 24):
                    e = PlanarEvaluator(c["kind"], 3, [Request(**r) for r in c["requests"]],
                        kernel=kernel, method="shell_quadrature", nodes=nodes, precision=precision, device=device)
                    values.append(e.evaluate(float(c["distance"]), c["normal"], 1,
                        **({} if c["half_width"] is None else dict(half_widths=float(c["half_width"]))))[0])
                ref = np.array(c["values"], float)
                errors.extend((abs(values[0]-ref)/(1+abs(ref))).tolist())
                convergence.extend((abs(values[0]-values[1])/(1+abs(values[1]))).tolist())
            errors = np.array(errors)
            row = dict(precision=precision, kernel=kernel, cases=count, channels=len(errors), budget=budget,
                worst_mixed_error=float(errors.max()), failing_channels=int(np.sum(~np.isfinite(errors)|(errors>budget))),
                radial_nodes=16, convergence_nodes=24, worst_mixed_convergence=max(convergence),
                rejected_cases=0, fallback_count=0)
            rows.append(row)
            if row["failing_channels"] or row["worst_mixed_convergence"] > budget: raise RuntimeError(row)
    return rows


def runtime(device):
    rows = []; references = {}
    jets = [()] + [(i,) for i in range(3)] + [(i,j) for i in range(3) for j in range(i,3)]
    masks = dict(baseline=baseline(3), degree9=[Request(a,ds) for a in monomials(3,9) for ds in jets],
                 degree9_moments=[Request(a) for a in monomials(3,9)])
    for kind in ("plane", "strip"):
        for mask, requests in masks.items():
            for method, precision in (("analytic", "float64"), ("shell_quadrature", "float64"), ("shell_quadrature", "float32")):
                for count in ((128,) if mask == "baseline" else ((1,32) if method == "shell_quadrature" and device.is_cuda else (1,))):
                    distances = np.linspace(-.35,.9,count); normal = [0,0,1]
                    kwargs = {} if kind == "plane" else dict(half_widths=.2)
                    ev, prep = timed(lambda: PlanarEvaluator(kind, 3, requests, kernel="w2", method=method,
                        precision=precision, device=device), device, 1)
                    batch, allocate = timed(lambda: ev.prepare(distances, normal, 1, **kwargs), device, 3)
                    _, first = timed(batch.launch, device, 1)
                    _, warm = timed(batch.launch, device, 3, events=True)
                    _, complete = timed(lambda: ev.evaluate(distances, normal, 1, **kwargs), device, 1)
                    _, readback = timed(batch.numpy, device, 3)
                    row = dict(kind=kind, mask=mask, method=method, precision=precision, queries=count,
                        outputs=len(requests), kernel="w2", batch_bytes=batch.bytes, plan_preparation=prep,
                        allocation_upload=allocate, first_launch=first, warm=warm, complete=complete,
                        readback=readback, graph_replay=None)
                    key = (kind, mask, count)
                    if method == "analytic": references[key] = batch.numpy()
                    ref = references.get(key, references.get((kind, mask, 1)))
                    compared = batch.numpy() if ref.shape[0] == count else batch.numpy()[:1]
                    difference = float(np.max(abs(compared-ref)/(1+abs(ref))))
                    row["mixed_difference_from_matched_analytic_fp64"] = difference
                    row["matched_analytic_comparison_queries"] = ref.shape[0]
                    if difference > (1e-10 if precision == "float64" else 1e-5):
                        raise RuntimeError("matched quadrature/analytic outputs differ")
                    if device.is_cuda:
                        with wp.ScopedCapture(device=device) as capture: batch.launch()
                        before = batch.numpy(); wp.capture_launch(capture.graph)
                        row["graph_bitwise_equal"] = bool(np.array_equal(before, batch.numpy()))
                        if not row["graph_bitwise_equal"]: raise RuntimeError("graph output changed")
                        _, row["graph_replay"] = timed(lambda: wp.capture_launch(capture.graph), device, 3, events=True)
                    rows.append(row)
                    print(kind, mask, method, precision, count, warm["median_ms"], flush=True)
    return rows


def compile_only(precision, output, arch):
    import warp._src.build as build
    PlanarEvaluator("plane", 3, method="shell_quadrature", precision=precision)
    module = wp.get_module("warpSPHBoundaries.fluxbridge.planar_shell")
    options = dict(module.options); options["block_dim"] = 256
    source, extension, _, ltoirs, fatbins = module._run_codegen(options, is_cpu=False)
    path = output/f"shell-{precision}.{extension}"; path.write_text(source)
    ptx = output/f"shell-{precision}.ptx"
    build.build_cuda(str(path), arch, str(ptx), pch_dir=None, use_precompiled_headers=False,
        fast_math=False, fuse_fp=False, ltoirs=ltoirs, fatbins=fatbins)
    instructions = [s.strip() for s in ptx.read_text().splitlines() if ".f64" in s and not s.strip().startswith((".", "//"))]
    if precision == "float32" and instructions: raise RuntimeError("fp64 in raw-fp32 kernel")
    report = dict(precision=precision, arch=arch, execution_tested=False, compile_passed=True,
        fp64_instructions=len(instructions), source_sha256=sources())
    (output/f"compile-{precision}.json").write_text(json.dumps(report, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--accuracy-only", action="store_true")
    p.add_argument("--compile-only", choices=["float32","float64"]); p.add_argument("--arch", type=int, default=120)
    args = p.parse_args(); args.output.mkdir(exist_ok=True, parents=True); wp.init()
    if args.compile_only: compile_only(args.compile_only, args.output, args.arch); return
    device = wp.get_device(args.device)
    with (open("/tmp/fluxbridge-gpu.lock", "w") if device.is_cuda else nullcontext(None)) as lock:
        if lock is not None: fcntl.flock(lock, fcntl.LOCK_EX)
        report = dict(schema=1, timestamp_utc=datetime.now(timezone.utc).isoformat(), warp=wp.__version__,
            device=str(device), device_name=device.name, cuda_runtime_tested=device.is_cuda,
            source_sha256=sources(), accuracy=accuracy(device), runtime=[] if args.accuracy_only else runtime(device),
            limitations=["3D infinite planes/bands only; no arbitrary tetrahedron or general material replacement",
                "near-roundoff finite fixture evidence; no universal relative error guarantee",
                "cubic outer Hessian has Laurent terms; radial Gauss is not polynomial exact there",
                "shared GPU; host-synchronized timing and CUDA-event intervals reported separately"])
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(args.output/"report.json")


if __name__ == "__main__": main()
