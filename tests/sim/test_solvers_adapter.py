"""The scheme-independent interface of the example notebooks (`sim/solvers.py`): the same flow description runs on both solvers and reproduces the closed-form references; the example notebooks are valid
notebooks with the documented switches, and the fast one (Taylor-Green vortex) runs end to end for both schemes."""
import json
import math
import pathlib

import numpy as np
import pytest
import torch
import warp as wp

import warpSPHBoundaries  # noqa: F401
from warpSPHBoundaries.scene.periodic import Periodic
from warpSPHBoundaries.sim.references import couette_cylinders, poiseuille, sangani_acrivos, slosh_omega, tgv_energy
from warpSPHBoundaries.sim.solvers import SCHEMES, box_lattice, lattice, make_solver, wall_gap

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")
EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "examples"


def test_references_closed_forms():
    assert tgv_energy(0.0, 0.01) == 1.0 and abs(tgv_energy(1.0, 0.01) - math.exp(-4 * 0.01 * (2 * math.pi) ** 2)) < 1e-15
    assert abs(poiseuille(0.25, 0.5, 0.1, 0.02) - 0.1 * 0.25 * 0.25 / 0.04) < 1e-15
    u, A, B, T = couette_cylinders(0.2, 0.2, 0.5, 2.0, nu=0.01)
    assert abs(u - 2.0 * 0.2) < 1e-14 and abs(T + 4 * math.pi * 0.01 * B) < 1e-15                      # the inner wall moves with the cylinder
    assert abs(couette_cylinders(0.5, 0.2, 0.5, 2.0)[0]) < 1e-14                                      # the outer wall is at rest
    assert abs(sangani_acrivos(0.1) - 4 * math.pi / (-0.5 * math.log(0.1) - 0.738 + 0.1 - 0.00887 + 0.002038)) < 1e-12
    assert abs(slosh_omega(1.0, 0.5) ** 2 - 9.81 * math.pi * math.tanh(math.pi * 0.5)) < 1e-12


@pytest.mark.parametrize("scheme", SCHEMES)
def test_box_lattice_gaps(scheme):
    dx = 1.0 / 16
    pos, hi = box_lattice(scheme, (0.0, 0.0), (8, 6), dx, walls=(False, True))
    g = wall_gap(scheme, dx)
    assert abs(pos[:, 1].min() - g) < 1e-12 and abs(hi[1] - pos[:, 1].max() - g) < 1e-12              # the first and last rows are a wall gap from the walls
    assert abs(pos[:, 0].min() - 0.5 * dx) < 1e-12 and abs(hi[0] - 8 * dx) < 1e-12                    # the periodic axis: cell centres, length n dx


@pytest.mark.parametrize("scheme", SCHEMES)
def test_taylor_green_decay_on_both_schemes(scheme):
    """the unified interface reproduces the TGV decay rate of the viscous operator to 3 % on a 32^2 lattice (measured 2.3 % delta, 0.4 % dfsph)."""
    n, nu, U = 32, 0.0185, 0.1
    dx = 1.0 / n
    pos = lattice((0, 0), (1, 1), dx)
    k = 2 * math.pi
    vel = U * np.stack([-np.cos(k * pos[:, 0]) * np.sin(k * pos[:, 1]), np.sin(k * pos[:, 0]) * np.cos(k * pos[:, 1])], 1)
    s = make_solver(scheme, pos, dx, None, nu=nu, vel=vel, periodic=Periodic((0, 0), (1, 1)), c0=10 * U, **({"maxDt": 5e-3} if scheme == "dfsph" else {}))
    ke = lambda: 0.5 * float((s.volume * (s.v ** 2).sum(1)).sum())
    E0 = ke()
    s.run(1.0)
    rate = -math.log(ke() / E0) / s.time
    assert abs(rate / (4 * nu * k * k) - 1.0) < 0.03, rate / (4 * nu * k * k)


def test_example_notebooks_are_wellformed():
    nbs = sorted(EXAMPLES.glob("*.ipynb"))
    assert len(nbs) >= 8
    for p in nbs:
        nb = json.loads(p.read_text())
        src = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
        assert 'SCHEME = "' in src[0] and 'QUALITY = "' in src[0] and "DEVICE" in src[0], p.name       # the switches of the first code cell
        assert any(c["cell_type"] == "markdown" for c in nb["cells"][:1]), p.name


@pytest.mark.slow
@pytest.mark.parametrize("scheme", SCHEMES)
def test_taylor_green_notebook_runs(scheme):
    """executes notebook 01 (about 10 s) with the scheme switched."""
    import re
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(EXAMPLES / "01_taylor_green_vortex.ipynb", as_version=4)
    for c in nb.cells:
        if c.cell_type == "code":
            c.source = re.sub(r'^SCHEME\s*=.*$', f'SCHEME = "{scheme}"', c.source, count=1, flags=re.M)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(EXAMPLES)}}).execute()
