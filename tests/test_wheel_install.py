"""The built wheel works without the checkout (docs/plan-next-steps.md "Remaining items" 9).

The wheel is built from a COPY of the sources (no build/ or egg-info in the working tree) and installed with `--no-deps` into a throw-away venv that only sees the running interpreter's
site-packages (torch, warp, warpSPHCore) as a fallback: nothing is installed into the environment the tests run in.  A child process started from an empty directory (no pyproject.toml above it, so
`paths.REPO_ROOT` is None) imports `warpSPHBoundaries.scene`, checks that it comes from the venv with the packaged tier-3 tables, and evaluates the fused wall aggregate of one disk; the numbers equal
the in-process (checkout) evaluation to 1e-12 (same code, same float64 kernels).
Marked slow: builds a wheel and spawns two interpreters (about a minute).
"""
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries
from warpSPHBoundaries import paths
from warpSPHBoundaries.edge.precision import real

pytestmark = [pytest.mark.slow, pytest.mark.skipif(real != wp.float64 or not wp.is_cuda_available() or paths.REPO_ROOT is None, reason="float64 contracts, GPU, needs the checkout")]

PROBE = textwrap.dedent('''
    import json, math, sys
    import numpy as np, torch
    import warpSPHBoundaries
    from warpSPHBoundaries import paths
    from warpSPHCore import KernelFunctions, ParticleState
    from warpSPHBoundaries.scene import AnalyticBoundary, Body, ImplicitRep, Scene
    from warpSPHBoundaries.scene.implicitBodies import DiskBody

    def evaluate(dev="cuda:0"):
        H = 0.3
        scene = Scene([Body(bodyId=0, center=(0.0, 0.0), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=0.2))])], dev)
        pos = torch.tensor([[0.3, 0.0], [0.0, 0.35], [-0.25, -0.2], [0.5, 0.5]], dtype=torch.float64, device=dev)
        n = len(pos)
        ps = ParticleState(positions=pos, supports=torch.full((n,), H, dtype=torch.float64, device=dev), masses=torch.ones(n, dtype=torch.float64, device=dev),
                           kinds=torch.zeros(n, dtype=torch.int32, device=dev), densities=torch.ones(n, dtype=torch.float64, device=dev))
        ag = AnalyticBoundary(scene).aggregate(ps, H, KernelFunctions.Wendland2, laplacian=True)
        return {k: v.sum(0).cpu().numpy().tolist() for k, v in ag.out.items()}

    if __name__ == "__main__":
        print(json.dumps(dict(file=warpSPHBoundaries.__file__, root=None if paths.REPO_ROOT is None else str(paths.REPO_ROOT), tables=str(paths.packaged_tables_dir()), out=evaluate())))
''')


def test_the_wheel_installs_and_evaluates_a_disk_without_the_checkout(tmp_path):
    root = paths.REPO_ROOT
    src = tmp_path / "src_copy"
    src.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(root / name, src / name)
    shutil.copytree(root / "src", src / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc"))
    wheels = tmp_path / "wheels"
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-q", "-w", str(wheels), str(src)], check=True, capture_output=True, text=True)
    (wheel,) = list(wheels.glob("*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    assert any(n.startswith("warpSPHBoundaries/data/tables/") and n.endswith(".npz") for n in names), "the tier-3 tables must ship in the wheel"
    assert not any(n.startswith("tests/") or "/tests/" in n for n in names)

    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True, capture_output=True, text=True)
    py = venv / "bin" / "python"
    subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--ignore-installed", "-q", str(wheel)], check=True, capture_output=True, text=True)

    empty = tmp_path / "empty"
    empty.mkdir()
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path / "cache"), "WARPSPHBOUNDARIES_TMP": str(tmp_path / "scratch")}
    r = subprocess.run([str(py), str(probe)], cwd=empty, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert str(venv) in got["file"], got["file"]                                              # the venv's copy, not the checkout's editable one
    assert got["root"] is None                                                               # no checkout above the child's package
    assert Path(got["tables"]).is_dir() and list(Path(got["tables"]).glob("*.npz"))

    ns = {}
    exec(compile(PROBE, "probe", "exec"), ns)
    ref = ns["evaluate"]()
    assert set(got["out"]) == set(ref)
    for k in ref:
        a, b = np.asarray(got["out"][k]), np.asarray(ref[k])
        assert np.abs(a - b).max() <= 1e-12 * max(np.abs(b).max(), 1e-12), k
    assert np.abs(np.asarray(ref["lam"])).max() > 0                                          # the disk is within the support of the queries: a non-trivial aggregate
