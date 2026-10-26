"""warpSPHBoundaries -- exact 2D SPH boundary integrals via divergence-theorem edge reductions.

Sub-packages: `curvbound` (closed forms and quadrature oracles of the PLAN track), `edge` (the edge machinery), `mesh3d` (analytic tetrahedral/surface reference), `scene` (bodies, boundary operations, wall operators, the boundary provider); `paths`.  The DFSPH / delta+-SPH reference solvers (`sim`) were retired into warpSPH on 2026-10-11 (git tag oracle-solvers-2026-10-11).
Derivations: docs/derivations/*.md.  The 2D edge reductions (`edge`) and the PLAN-track closed forms (`curvbound`) are separate tracks that share the kernels.
"""
__version__ = "0.1.5"          # keep equal to pyproject.toml (scripts/publish_pypi.sh checks it)

from .edge.precision import ensure_default as _ensure_default      # warpSPHCore precision: warpSPHBoundaries defaults to float64 unless configured (edge/precision.py); before any warpSPHCore import
_ensure_default()

from .edge.core import gradient, moment, moment_gradient, value
from .edge.geometry import prepare
from .edge.kernels import KERNELS, disk_moment, kernel
from .edge.oracle import polar_disk_gradient, polar_disk_moment, polar_disk_value, polar_gradient, polar_moment, polar_value

__all__ = ["value", "gradient", "moment", "moment_gradient", "prepare",
           "KERNELS", "kernel", "disk_moment",
           "polar_value", "polar_gradient", "polar_moment", "polar_disk_value", "polar_disk_moment", "polar_disk_gradient"]
