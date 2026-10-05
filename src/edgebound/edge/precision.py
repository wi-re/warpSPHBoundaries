"""Precision of the Warp edge kernels: the one mechanism of warpSPH, `warpSPHCore.type_config`.

warpSPHCore fixes `scalar_t` (and `vec_t`) once per process, before it is imported: environment variable `warpSPHCore_PRECISION` (float16 | float32 | float64) or
`warpSPHCore_config.configure(precision=..., dim=...)`; its own default is float32.  Every kernel of `warp2d` / `warpbc` is written against `real` / `vec2_t` below, i.e.
against that same `scalar_t`, so the edge machinery follows the precision of the SPH solver it is coupled to: nothing here sets a second, independent precision.

Two rules of this package on top of it:

* **edgebound's default is float64** (the validation, the exactness tests and the bit-level regression harness are float64 contracts).  `edgebound/__init__.py` calls
  `ensure_default()` before anything imports warpSPHCore; if nothing was configured it selects float64.  To run in float32 set `warpSPHCore_PRECISION=float32` in the
  environment (or `warpSPHCore_config.configure(precision="float32")` before the first import); if warpSPHCore was imported first with its float32 default, the edge kernels
  are float32 as well (a warning says so).
* **float32 uses the stable (Chebyshev) plans.**  The monomial plan amplifies float32 rounding 1e2 - 2e4x (docs/exactness-and-approximations.md), so `edge_channels` selects the
  Chebyshev route `stable=(8, 6)` by default in float32 (value 8e-8, gradient 3e-7 relative); the finite-element pair kernels (`pair_weights`, `warp2d.moment`) are float64 only.

Host-side plan compilation (exact rationals) stays float64 and is rounded once into the arrays of the active precision; the tensors the wrappers return are float64 (holding
values computed in `real`).
"""
import os
import sys
import warnings

import numpy as np
import warp as wp


def ensure_default():
    """select float64 as the warpSPHCore precision unless the user configured one (environment variable or `warpSPHCore_config`) or warpSPHCore is already imported."""
    if "warpSPHCore" in sys.modules or os.environ.get("warpSPHCore_PRECISION"):
        return
    try:
        import warpSPHCore_config as cfg
    except ModuleNotFoundError:
        return
    if cfg.precision is None:
        cfg.configure(precision="float64", dim=cfg.dim)


ensure_default()

from warpSPHCore.type_config import get_torch_precision, scalar_t as real      # noqa: E402  (after ensure_default)

import torch  # noqa: E402

IS_F64 = real == wp.float64
IS_F32 = real == wp.float32
vec2_t = wp.types.vector(length=2, dtype=real)
vec3_t = wp.types.vector(length=3, dtype=real)
np_real = {wp.float16: np.float16, wp.float32: np.float32, wp.float64: np.float64}[real]
torch_real = get_torch_precision()

if not IS_F64 and "edgebound_precision_warned" not in os.environ:
    os.environ["edgebound_precision_warned"] = "1"
    warnings.warn("edgebound edge kernels run in %s (warpSPHCore.type_config.scalar_t); the exactness tests, the regression harness and the finite-element pair kernels "
                  "are float64 contracts (set warpSPHCore_PRECISION=float64)" % real.__name__, RuntimeWarning, stacklevel=2)


def require_f64(what):
    if not IS_F64:
        raise NotImplementedError("%s is float64 only (active precision %s): set warpSPHCore_PRECISION=float64" % (what, real.__name__))
