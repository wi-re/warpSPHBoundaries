"""The package layering of the boundary library (the future `warpSPHBoundaries`): `edge` <- `scene` <- `sim`.  `edge` and `scene` may import `warpSPHCore` (particle state, operation vocabulary, precision) but never
`warpSPH` and never a higher layer, so that the scene layer installs and runs without the scheme library; the public scene surface exists and the analytic provider satisfies the `BoundaryProvider` protocol.
"""
import ast
from pathlib import Path

import pytest
import torch

import warpSPHBoundaries
from warpSPHBoundaries import scene as scene_pkg

SRC = Path(warpSPHBoundaries.__file__).parent


def imports(path):
    """(module names, relative-import levels) imported anywhere in the file, as dotted strings relative to the package."""
    pkg = path.relative_to(SRC).with_suffix("").parts[:-1]
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(pkg[:len(pkg) - (node.level - 1)]) if node.level > 1 else list(pkg)
                out.append("warpSPHBoundaries." + ".".join(base + ([node.module] if node.module else [])))
            else:
                out.append(node.module or "")
    return out


@pytest.mark.parametrize("layer,forbidden", [("edge", ("warpSPHBoundaries.scene", "warpSPHBoundaries.sim", "warpSPH.", "warpSPH")), ("scene", ("warpSPHBoundaries.sim", "warpSPH.", "warpSPH"))])
def test_no_upward_imports(layer, forbidden):
    bad = []
    for f in sorted((SRC / layer).glob("*.py")):
        for m in imports(f):
            if m == "warpSPH" or m.startswith("warpSPH.") or any(m == fb.rstrip(".") or m.startswith(fb.rstrip(".") + ".") for fb in forbidden if fb.startswith("warpSPHBoundaries")):
                bad.append((f.name, m))
    assert not bad, bad


def test_public_scene_surface_and_provider_protocol():
    for name in scene_pkg.__all__:
        assert getattr(scene_pkg, name) is not None, name
    from warpSPHBoundaries.scene import AnalyticBoundary, BoundaryProvider, Body, DiskBody, ImplicitRep, Scene
    assert {"Scene", "Body", "AnalyticBoundary", "BoundaryProvider", "WallAggregate", "SdfRep", "ImplicitRep"} <= set(scene_pkg.__all__)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    sc = Scene([Body(bodyId=0, reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=0.3))])], dev)
    prov = AnalyticBoundary(sc)
    assert isinstance(prov, BoundaryProvider) and prov.bodies is sc.bodies and prov.supported()
    with pytest.raises(AttributeError):
        scene_pkg.NoSuchThing


def test_tabulated_kernels_ship_with_the_package():
    """the tier-3 tables are package data (an installed wheel does not rebuild them: ~20 s per kernel with mpmath), and the build configuration lists them."""
    from warpSPHBoundaries import paths
    names = {p.name for p in paths.packaged_tables_dir().glob("tier3_*.npz")}
    assert {"tier3_w2.npz", "tier3_w4.npz"} <= names
    pyproject = (SRC.parents[1] / "pyproject.toml").read_text() if (SRC.parents[1] / "pyproject.toml").exists() else None
    if pyproject is not None:
        assert '"data/tables/*.npz"' in pyproject and '"data/symbolic/*.txt"' in pyproject
    assert {p.name for p in (SRC / "data" / "symbolic").glob("*_export.txt")} == {"planar_export.txt", "sphere_export.txt"}          # the Maple exports of warpSPHBoundaries.curvbound
