"""edgebound -- exact 2D SPH boundary integrals via divergence-theorem edge reductions.

Stage 1 (mpmath) reference implementation of HANDOFF.md tiers 1/2 (2D).
Derivations: docs/derivations/*.md.  Do not confuse with `curvbound` (PLAN track).
"""
from .core import gradient, moment, moment_gradient, value
from .geometry import prepare
from .kernels import KERNELS, disk_moment, kernel
from .oracle import polar_gradient, polar_moment, polar_value

__all__ = ["value", "gradient", "moment", "moment_gradient", "prepare",
           "KERNELS", "kernel", "disk_moment",
           "polar_value", "polar_gradient", "polar_moment"]
