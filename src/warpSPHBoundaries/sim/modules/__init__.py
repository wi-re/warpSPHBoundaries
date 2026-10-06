"""Warp modules of the fluid side that warpSPH does not have (yet), in warpSPH's layout: `modules/<task>/wp_<function>.py` holds the kernels in the template of warpSPH's own modules
(`<fn>_Func_i` neighbour loop, `<fn>_Func_Adjacency` adjacency / grid traversal, `<fn>_Kernel` and `<fn>_KernelTiled`, an `OperatorSpec`, and the torch-facing `<fn>Warp(queryParticles, operationProperties, domain, ...)`
wrapper on `launchOperator`), so a module moves into warpSPH by copying the file.  They are PARTIAL SUMS of the free-surface detector and of the shifting: what a boundary representation (analytic integrals,
wall particles) has to add its own contribution to before the decision / the surface treatment.  What warpSPH already has is used from there (`dilateSurfaceMaskWarp`, `warpOperation` Covariance / Gradient,
`countNeighbors`); see docs/audit-warpsph-boundary-hooks.md §6 for the comparison.  Generated from one template by docs/work/refs/gen_wp_modules.py (kept in the repository for the next module).
"""
