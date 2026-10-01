# curvatureBoundaries

Curvature-aware closed forms for the semi-analytic SPH boundary integral of
Winchenbach, Akhunov & Kolb (2020),

```
lambda(x) = Int_{D_B(x)} W(|x-x'|, h) dx',     W(r,h) = C_n/h^n W_hat(r/h)
```

extended beyond the paper's 3-D planar cubic-spline result to:

* **2-D planar** boundaries for all four kernels — new arccos/ln closed forms
  (`maple/01_planar.mpl`, §3 of `docs/derivation.md`);
* **3-D planar** Wendland kernels w2/w4/w6 — single polynomials of degree 2n
  via the monomial basis `F(n,p,x)`;
* **curved boundaries**: solid sphere for the Wendland kernels — both
  branches (`2R+d >= 1` and `2R+d < 1`) are rational `N(R,d)/(R+d)`, with
  exact planar limit `R -> infinity`.

Kernels and conventions follow warpSPHCore
(`src/warpSPHCore/kernels/`, `scripts/kernels/kernel_specs.yaml`):
`q = r/h` with `h` the support radius.

## Layout

```
maple/00_setup.mpl          kernel table, normalization, F(n,p,x) basis checks
maple/01_planar.mpl         2-D + 3-D planar, all kernels
maple/03_sphere.mpl         solid sphere (Wendland w2/w4/w6)
maple/run_all.sh            regenerate results/symbolic/ (needs Maple 2026)
python/curvbound/           kernel table, closed-form evaluators, oracles
tests/                      pytest suite (49 tests)
notebooks/demo.ipynb        executed demo: geometry, curves, validation
docs/derivation.md          full mathematical write-up
PLAN.md                     project status & roadmap
results/symbolic/           Maple exports (committed; consumed by python/)
results/figures/            notebook figures (generated, git-ignored)
```

## Quick start

```bash
# 1. (re)generate the symbolic results — needs Maple 2026
maple/run_all.sh                       # MAPLE=/path/to/maple to override

# 2. run the validation suite (any python with mpmath + pytest)
python -m pytest tests/ -q

# 3. run the demo notebook (jupyter + matplotlib)
python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/demo.ipynb --ExecutePreprocessor.kernel_name=python3
```

Python API sketch:

```python
import sys; sys.path.insert(0, "python")
from fractions import Fraction
from curvbound import planar2d, planar3d, sphere

planar3d("w4", Fraction(1, 3))          # exact rational (polynomial in d)
planar2d("w4", "0.3")                   # mpf (arccos/ln closed form)
sphere("w4", Fraction(4, 10), Fraction(1, 5))   # solid ball R=0.4, d=0.2
```

## Validation

Every closed form is validated three ways (details in `docs/derivation.md`
§5): exact symbolic identities in Maple, 50-digit Maple quadrature, and
independent 40-digit mpmath oracles in `python/curvbound/oracle.py`
(including a true 2-D Cartesian quadrature and the h-scaling identity).
Worst observed error: ~1e-39.

## Status

3-D planar (cubic + Wendland) and 3-D spherical (Wendland) were completed in
the original plan; the 2-D derivations, the promotion of the toy harnesses
to this repository layout, and the demo notebook are done (see `PLAN.md`
§18–§18c). Next: cylinder boundary, 2-D circle boundary, inside-solid
(`d < 0`) cases, gradient terms.
