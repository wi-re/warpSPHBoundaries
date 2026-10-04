"""Tests for WORK-006: the exact wall operations (cover, cone, tensile) are the only path, and the wall viscosity is
selected by cfg.wallViscosityForm ("laplacian" default, "pairwise" kept).

  (i)    DeltaSPHConfig has none of the removed switches, keeps surfaceSamples, and wallViscosityForm = "laplacian".
  (ii)   a bogus wallViscosityForm raises ValueError in rhs (matching the form name).
  (iii)  three sim.step(): Scene.inside is called 0 times with the default ("laplacian") form and > 0 with "pairwise"
         (the polar grid is built only for the pairwise wall term); Scene.buildAdjacency per step in the default form
         is also counted (information).
"""
import pytest
import warp as wp

from edgebound.deltasph2d import DeltaSPHConfig, hydrostatic_tank
from edgebound.scene import Scene

DEVICES = ["cuda:0"] if wp.is_cuda_available() else ["cpu"]


def test_removed_switches_are_gone():
    """(i) the four exact-operation switches are gone; surfaceSamples remains and wallViscosityForm = "laplacian"."""
    cfg = DeltaSPHConfig()
    for name in ("coverExact", "coneExact", "tensileExact", "viscosityExact"):
        assert not hasattr(cfg, name), name
    assert hasattr(cfg, "surfaceSamples")
    assert cfg.wallViscosityForm == "laplacian"


@pytest.mark.parametrize("device", DEVICES)
def test_bogus_wall_viscosity_form_raises(device):
    """(ii) wallViscosityForm = "bogus" raises ValueError (matching the form name) in rhs on the near-wall particles."""
    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device)
    sim.cfg.wallViscosityForm = "bogus"
    with pytest.raises(ValueError, match="wallViscosityForm"):
        sim.rhs(sim.x, sim.v, sim.rho)
    sim.cfg.wallViscosityForm = "laplacian"


@pytest.mark.parametrize("device", DEVICES)
def test_inside_calls_by_form(device):
    """(iii) three sim.step(): Scene.inside called 0 times with the default ("laplacian") form, > 0 with "pairwise";
    Scene.buildAdjacency per step in the default form is printed (information)."""
    counts = {"inside": 0, "build": 0}
    orig_inside = Scene.inside
    orig_build = Scene.buildAdjacency

    def counting_inside(self, *a, **k):
        counts["inside"] += 1
        return orig_inside(self, *a, **k)

    def counting_build(self, *a, **k):
        counts["build"] += 1
        return orig_build(self, *a, **k)

    sim, _ = hydrostatic_tank(dp=0.04, domain="surface", device=device)
    Scene.inside = counting_inside
    Scene.buildAdjacency = counting_build
    try:
        counts["inside"] = 0
        counts["build"] = 0
        for _ in range(3):
            sim.step()                                              # default form: "laplacian"
        n_inside_default, n_build_default = counts["inside"], counts["build"]
        counts["inside"] = 0
        sim.cfg.wallViscosityForm = "pairwise"
        for _ in range(3):
            sim.step()                                              # pairwise form: the polar grid is built
        n_inside_pairwise = counts["inside"]
    finally:
        Scene.inside = orig_inside
        Scene.buildAdjacency = orig_build
        sim.cfg.wallViscosityForm = "laplacian"
    assert n_inside_default == 0, n_inside_default
    assert n_inside_pairwise > 0, n_inside_pairwise
    print("(iii) default (laplacian) form: Scene.inside calls over 3 steps = %d (expect 0); Scene.buildAdjacency = %d (information)"
          % (n_inside_default, n_build_default))
    print("(iii) pairwise form: Scene.inside calls over 3 steps = %d (expect > 0)" % n_inside_pairwise)