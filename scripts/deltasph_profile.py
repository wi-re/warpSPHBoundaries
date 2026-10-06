"""Profile one DeltaSPH2D.step: where does the time go?  (WORK-001 T0.2)

Monkey-patches the listed callables with a stack-based timer (torch.cuda.synchronize() before
and after each call) so that BOTH inclusive and self times are reported correctly for nested
calls (a parent's self time excludes its tracked children).  The patched callables are restored
afterwards; the solver is NOT modified.

Cases: dam break (marrone_dambreak nx=67, shifting=True, noPen=impulse): 300 warm-up + 200 timed
steps.  Sloshing (sloshing_tank nx=200, shifting=True, noPen=impulse): 100 warm-up + 100 timed.
Also a torch.profiler run of 20 dam-break steps (top 15 by CUDA time and by CPU time).

Run:  python scripts/deltasph_profile.py
"""
import time

import numpy as np
import torch

import warpSPHBoundaries.sim.deltasph2d as d2d
from warpSPHBoundaries.sim.deltasph2d import DeltaSPH2D
from warpSPHBoundaries.sim.cases import marrone_dambreak, sloshing_tank
from warpSPHBoundaries.scene.scene import Scene, SceneAdjacency
from warpSPHBoundaries.scene.fused import FusedWall
from warpSPHBoundaries import paths

# --------------------------------------------------------------------------- stack-based timer
_STATS = {"calls": {}, "inclusive": {}, "self": {}}
_STACK = []
_INSIDE_POINTS = 0
_PARENT_CHILD = {}   # parent name -> {child name -> cumulative inclusive time of that child called from the parent}


def _enter(name):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if _STACK:
        p = _STACK[-1]
        _STATS["self"][p["name"]] = _STATS["self"].get(p["name"], 0.0) + (t0 - p["last"])
        p["last"] = t0
    _STATS["calls"][name] = _STATS["calls"].get(name, 0) + 1
    _STACK.append({"name": name, "entry": t0, "last": t0})


def _exit(name):
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    f = _STACK.pop()
    assert f["name"] == name, ("stack mismatch", f["name"], name)
    inc = t1 - f["entry"]
    _STATS["self"][name] = _STATS["self"].get(name, 0.0) + (t1 - f["last"])
    _STATS["inclusive"][name] = _STATS["inclusive"][name] if name in _STATS["inclusive"] else 0.0
    _STATS["inclusive"][name] += inc
    if _STACK:
        par = _STACK[-1]["name"]
        pc = _PARENT_CHILD.setdefault(par, {})
        pc[name] = pc.get(name, 0.0) + inc
        _STACK[-1]["last"] = t1


def _reset():
    _STATS["calls"].clear()
    _STATS["inclusive"].clear()
    _STATS["self"].clear()
    _STACK.clear()
    _PARENT_CHILD.clear()
    global _INSIDE_POINTS
    _INSIDE_POINTS = 0


# tracked names: (display name, kind, target, attr)
# kind "mod": a module-level function in the d2d namespace; kind "cls": a class method
_TRACK = [
    ("deltasph2d.neighbor_pairs", "mod", d2d, "neighbor_pairs"),
    ("deltasph2d.sceneOperation", "mod", d2d, "sceneOperation"),
    ("Scene.adjacency", "cls", Scene, "adjacency"),
    ("Scene.precompute", "cls", Scene, "precompute"),
    ("SceneAdjacency.restrict", "cls", SceneAdjacency, "restrict"),
    ("FusedWall (stage 1)", "cls", FusedWall, "__init__"),
    ("FusedWall.evaluate", "cls", FusedWall, "evaluate"),
    ("deltasph2d.cover_vector_scene", "mod", d2d, "cover_vector_scene"),
    ("deltasph2d.cone_area_scene", "mod", d2d, "cone_area_scene"),
    ("deltasph2d.lap_lambda_scene", "mod", d2d, "lap_lambda_scene"),
    ("deltasph2d.tensile_vector_scene", "mod", d2d, "tensile_vector_scene"),
    ("Scene.inside", "cls", Scene, "inside"),
    ("Scene.signed_distance", "cls", Scene, "signed_distance"),
    ("DeltaSPH2D._solid_samples", "cls", DeltaSPH2D, "_solid_samples"),
    ("DeltaSPH2D._detect_surface", "cls", DeltaSPH2D, "_detect_surface"),
    ("DeltaSPH2D._wall_data", "cls", DeltaSPH2D, "_wall_data"),
    ("DeltaSPH2D._surface_state", "cls", DeltaSPH2D, "_surface_state"),
    ("DeltaSPH2D.rhs", "cls", DeltaSPH2D, "rhs"),
    ("DeltaSPH2D.shift", "cls", DeltaSPH2D, "shift"),
    ("DeltaSPH2D.no_penetration", "cls", DeltaSPH2D, "no_penetration"),
    ("DeltaSPH2D.step", "cls", DeltaSPH2D, "step"),
]


def _wrap(name, orig, points_arg=None):
    def wrapper(*args, **kwargs):
        _enter(name)
        try:
            if points_arg is not None and len(args) > points_arg:
                global _INSIDE_POINTS
                _INSIDE_POINTS += int(args[points_arg].shape[0])
            return orig(*args, **kwargs)
        finally:
            _exit(name)
    return wrapper


def _patch():
    saved = []
    for name, kind, target, attr in _TRACK:
        orig = getattr(target, attr)
        # class methods take self as args[0], so the `points` of Scene.inside is args[1]
        pa = 1 if name == "Scene.inside" else None
        w = _wrap(name, orig, points_arg=pa)
        setattr(target, attr, w)
        saved.append((target, attr, orig))
    return saved


def _unpatch(saved):
    for target, attr, orig in reversed(saved):
        setattr(target, attr, orig)


# --------------------------------------------------------------------------- one case
ORDER = ["DeltaSPH2D.step", "DeltaSPH2D.rhs", "DeltaSPH2D.shift", "DeltaSPH2D.no_penetration",
         "DeltaSPH2D._wall_data", "DeltaSPH2D._surface_state", "DeltaSPH2D._detect_surface",
         "DeltaSPH2D._solid_samples", "deltasph2d.neighbor_pairs", "deltasph2d.sceneOperation",
         "Scene.adjacency", "Scene.precompute", "SceneAdjacency.restrict", "FusedWall (stage 1)", "FusedWall.evaluate", "deltasph2d.cover_vector_scene", "deltasph2d.cone_area_scene", "deltasph2d.lap_lambda_scene",
         "deltasph2d.tensile_vector_scene", "Scene.inside", "Scene.signed_distance"]


def _run_case(label, build, warm, timed):
    print("\n" + "=" * 78)
    print("CASE: %s   (%d warm-up, %d timed steps)" % (label, warm, timed))
    print("=" * 78)
    sim, info = build()
    N = len(sim.x)
    print("N particles = %d   dx = %.5f   H = %.5f   nbodies = %d" % (N, float(sim.dx), float(sim.H), sim.nb))
    saved = _patch()
    try:
        _reset()
        for _ in range(warm):
            sim.step()
        _reset()
        t_wall0 = time.perf_counter()
        for _ in range(timed):
            sim.step()
        torch.cuda.synchronize()
        t_wall = time.perf_counter() - t_wall0
        nstep = _STATS["calls"].get("DeltaSPH2D.step", 0)
        assert nstep == timed, ("step calls", nstep, timed)
        step_inc = _STATS["inclusive"]["DeltaSPH2D.step"]
        ms_step_incl = 1000.0 * step_inc / nstep
        ms_step_wall = 1000.0 * t_wall / nstep
        print("ms/step  (timer inclusive of step) = %.3f    (wall time / steps) = %.3f" % (ms_step_incl, ms_step_wall))
        sum_self = sum(_STATS["self"].get(n, 0.0) for n in ORDER)
        print("sum of self times of the %d tracked names = %.3f ms/step  ->  %.1f%% of the step inclusive time" %
              (len(ORDER), 1000.0 * sum_self / nstep, 100.0 * sum_self / step_inc))
        print("\n%-28s %10s %14s %14s %9s" % ("name", "calls/step", "inclusive ms/step", "self ms/step", "% of step"))
        for n in ORDER:
            c = _STATS["calls"].get(n, 0) / nstep
            inc = 1000.0 * _STATS["inclusive"].get(n, 0.0) / nstep
            sf = 1000.0 * _STATS["self"].get(n, 0.0) / nstep
            pct = 100.0 * _STATS["inclusive"].get(n, 0.0) / step_inc
            print("%-28s %10.3f %14.4f %14.4f %9.2f" % (n, c, inc, sf, pct))
        # answers
        ba = _STATS["calls"].get("Scene.adjacency", 0) / nstep
        sop = _STATS["calls"].get("deltasph2d.sceneOperation", 0) / nstep
        inside_pts = _INSIDE_POINTS / nstep
        inside_pct = 100.0 * _STATS["inclusive"].get("Scene.inside", 0.0) / step_inc
        # fluid pair sums = "rhs minus scene minus detector": rhs inclusive minus the _wall_data /
        # _solid_samples / _detect_surface time that is WITHIN rhs (per-parent), leaving neighbor_pairs + the torch sums.
        rhs_inc = _STATS["inclusive"].get("DeltaSPH2D.rhs", 0.0)
        rhs_pc = _PARENT_CHILD.get("DeltaSPH2D.rhs", {})
        scene_det = (rhs_pc.get("DeltaSPH2D._wall_data", 0.0) + rhs_pc.get("DeltaSPH2D._solid_samples", 0.0)
                     + rhs_pc.get("DeltaSPH2D._detect_surface", 0.0))
        fluid = rhs_inc - scene_det
        fluid_pct = 100.0 * fluid / step_inc
        costs = sorted(((1000.0 * _STATS["self"].get(n, 0.0) / nstep, n) for n in ORDER), reverse=True)
        print("\n(a) Scene.adjacency calls/step = %.3f ; sceneOperation calls/step = %.3f" % (ba, sop))
        print("(b) positions through Scene.inside per step = %.0f  ; Scene.inside = %.2f%% of the step" % (inside_pts, inside_pct))
        print("(c) fluid pair sums (rhs - scene - detector) = %.4f ms/step = %.2f%% of the step" % (fluid / nstep * 1000.0, fluid_pct))
        print("(e) three largest single costs (self ms/step): " + ";  ".join("%s = %.4f" % (n, ms) for ms, n in costs[:3]))
        return dict(label=label, N=N, dx=float(sim.dx), H=float(sim.H), nstep=nstep, step_inc=step_inc, wall=t_wall,
                    ms_step_incl=ms_step_incl, ms_step_wall=ms_step_wall, sum_self_ms=1000.0 * sum_self / nstep,
                    sum_self_pct=100.0 * sum_self / step_inc, ba=ba, sop=sop, inside_pts=inside_pts, inside_pct=inside_pct,
                    fluid_ms=fluid / nstep * 1000.0, fluid_pct=fluid_pct, costs=costs[:3])
    finally:
        _unpatch(saved)


def _profiler_dambreak(nsteps=20):
    print("\n" + "=" * 78)
    print("torch.profiler: %d dam-break steps (top 15 by CUDA time, then by CPU time)" % nsteps)
    print("=" * 78)
    sim, _ = marrone_dambreak(nx=67, shifting=True, noPen="impulse")
    for _ in range(5):
        sim.step()   # brief warm-up outside the profiler
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(nsteps):
            sim.step()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    print("\n--- top 15 by CUDA time (device time_total) ---")
    print(ka.table(sort_by="cuda_time_total", row_limit=15))
    print("\n--- top 15 by CPU time (cpu_time_total) ---")
    print(ka.table(sort_by="cpu_time_total", row_limit=15))


def main():
    torch.cuda.init()
    db = _run_case("dam break  (marrone_dambreak nx=67, shifting=True, noPen=impulse)",
                   lambda: marrone_dambreak(nx=67, shifting=True, noPen="impulse"), warm=300, timed=200)
    # (d) compare ms/step with the stored series wall/steps
    series = np.load(paths.results_dir() / "deltasph" / "dambreak_B_nx67_series.npz")
    s_wall, s_steps = float(series["wall"]), int(series["steps"])
    s_ms = 1000.0 * s_wall / s_steps
    print("\n(d) stored dambreak_B_nx67 series: wall = %.2f s, steps = %d  ->  %.3f ms/step" % (s_wall, s_steps, s_ms))
    print("    profiled dam-break ms/step (timer) = %.3f  ;  ratio profiled / series = %.3f" % (db["ms_step_incl"], db["ms_step_incl"] / s_ms))
    _run_case("sloshing  (sloshing_tank nx=200, shifting=True, noPen=impulse)",
                   lambda: sloshing_tank(nx=200, shifting=True, noPen="impulse"), warm=100, timed=100)
    _profiler_dambreak(nsteps=20)
    print("\nDONE.")


if __name__ == "__main__":
    main()
