"""warpSPHBoundaries.curvbound — curvature-aware SPH boundary integrals (the PLAN track; formerly the top-level package `curvbound`).

Closed forms (Maple-generated, data/symbolic/) plus independent
high-precision quadrature oracles, for the semi-analytic SPH boundary
handling of Winchenbach, Akhunov, Kolb (2020) extended to Wendland kernels
and to curved (solid-sphere) boundaries.
"""
from .kernels import KERNELS, Kernel, kernel
from .oracle import (
    physical_planar3d,
    quad_planar2d,
    quad_planar2d_cartesian,
    quad_planar3d,
    quad_sphere,
)
from .symbolic import PLANAR_EXPORT, SPHERE_EXPORT, planar2d, planar3d, sphere

__all__ = [
    "KERNELS", "Kernel", "kernel",
    "PLANAR_EXPORT", "SPHERE_EXPORT",
    "planar2d", "planar3d", "sphere",
    "quad_planar2d", "quad_planar2d_cartesian", "quad_planar3d",
    "quad_sphere", "physical_planar3d",
]
