# Validation, limitations and evidence reproduction

The repository retains mathematical derivations, API contracts, tests and
reproduction scripts. Development plans, progress logs, generated reports,
images and raw measurements live outside it, by default at:

```
../warpSPHDevelopment/curvatureBoundaries/
  progress/plan-2d-accuracy.md
  results/figures/
  logs/
  manifests/
```

The [development archive](../../warpSPHDevelopment/curvatureBoundaries/README.md)
contains the [acceptance matrix](../../warpSPHDevelopment/curvatureBoundaries/progress/plan-2d-accuracy.md)
and historical checkpoints. That sibling directory is local development data,
not a package dependency or a required checkout. Its Google Cloud backup is
manual. No cloud synchronization is configured by this project.

## Output configuration

Study commands default to the external project directory above. Set
`WARPSPH_DEVELOPMENT_ROOT` before running Python to choose another project
directory; `--output`, `--baseline` and `--reference-cache` override individual
paths. An installed wheel without a checkout defaults to
`~/dev/warpSPHDevelopment/curvatureBoundaries`.

Run these examples from the repository root:

```bash
export WARPSPH_DEVELOPMENT_ROOT="$(pwd)/../warpSPHDevelopment/curvatureBoundaries"
export EVIDENCE="$WARPSPH_DEVELOPMENT_ROOT/results/figures"
pip install -e '.[scripts,test]'
```

No generated figure or raw evidence array is required to run the tests. Tests
may reuse external reference caches and otherwise regenerate their references.
The shipped symbolic exports and kernel tables in `src/warpSPHBoundaries/data/`
are runtime package assets and remain in the repository.

## Documented accuracy scope

| Representation | Current contract and limitations | Reproducer |
|---|---|---|
| Explicit polygon / triangle | All twelve scalar/gradient moments in Scene/FusedWall. Small/sliver triangles use conditioned area quadrature; its estimates are empirical. Finite four-kernel gates do not establish universal few-double-eps accuracy or throughput. | `triangle_conditioning.py`, then `device_conditioning.py` |
| Sampled SDF guard | Rejects poorly conditioned differential fields over the entire support square. Requires an explicit accurate fallback for rejected supports. Acceptance is not an integral-error guarantee. | `sdf_guard.py` |
| Accepted smooth SDF | Scene exposes all twelve moments through one order-two curvature/variation expansion. Higher orders, differential reconstruction and geometric variation still limit accuracy. The tested smooth domain does not pass float32 on the complete twelve-moment contract. | `curvature_corrections.py` |
| Distance-only reference | All twelve local-half-plane reference moments; mass needs only distance, oriented moments need a normal. Corner, thin-feature and medial-axis geometry bias remains. | `sdf_distance_only.py` |
| Disk/box tables and specialized models | Restricted radius/width/curvature domains and moment coverage; no universal float32 approval. Geometry/model error must be distinguished from lookup/quadrature error. | `validation.robustness` |

The canonical order is `m00,m10,m01,m20,m11,m02,Gx,Gy,g1xx,g1xy,g1yx,g1yy`.
Use dimensionless support units, preserve unavailable entries as NaN, and keep
signed numerical, reconstruction and geometry/model contributions separate.
Report quadrature refinement and independent-reference uncertainty separately.
An order difference alone is not an error bound. Error plots distinguish values
above one, the epsilon32-to-one range, and epsilon64-to-epsilon32.

Detailed contracts: [surface moments](surface-moments.md),
[SDF guard](sdf-guard.md), [distance-only reference](sdf-distance-only.md),
[baseline accepted-SDF accuracy](sdf-accepted-accuracy.md) and
[corrected curvature moments](derivations/curvature-moments-2d.md), and
[exactness and approximations](exactness-and-approximations.md).

## Reproduce the important evidence

For a single shareable curved-wall figure comparing distance-only SDF,
curvature-corrected SDF and polygon refinement, run
`python scripts/studies/curved_boundary_example.py`. It shows boundary mass and
the normal first spatial moment against true-circle quadrature, with separate
error panels. PNG, vector SVG and raw measurements are written outside the repo.

For costs on that same geometry, run
`python scripts/studies/curved_boundary_cost.py --device cpu` or `--device cuda:0`.
It separates warm adjacency, moment evaluation, total cost and initial setup,
and records small/large batches with requested moments and all twelve channels.
The benchmark enforces float64 **kernel arithmetic**, records it separately
from returned tensor dtype, and rejects a process configured for float32.
`--sdf-backend torch --methods corrected_sdf --skip-arc` measures the same
current Scene interface using the pre-fusion Torch formulas as a baseline.
`--graph-route cheb` also measures the validated Chebyshev polygon route;
`--graph-route mono` selects analytic edge reduction. Both use float64.
The CUDA run also measures captured guarded-SDF core and fused polygon paths;
their scopes differ and are labelled. These are microbenchmarks, not solver
throughput or equal-accuracy comparisons. Raw repetitions and hardware details
are saved outside the repository.

The accepted smooth-SDF audit is standalone; it constructs actual curved-domain
references and independent high-precision boundary checks:

```bash
python scripts/studies/curvature_corrections.py --device cpu
python scripts/studies/curvature_corrections.py --device cuda:0 \
  --output "$EVIDENCE/curvature-corrections-fused-cuda" \
  --reference-cache "$EVIDENCE/sdf-accuracy-cpu" \
  --compare-to "$EVIDENCE/curvature-corrections-fused-cpu"
```

`--verify-backend-from "$EVIDENCE/curvature-corrections-cpu"` re-executes the
earlier checkpoint's queries with the current backend, retaining its validated
curved-domain references and shape-model budgets. Write to a distinct output
directory to preserve the baseline. Reports include backend output changes
and fallback-selection changes in `backend-verification.json`.

The polygon-derived studies share an expensive high-precision reference stage.
Generate it before running its consumers:

```bash
python scripts/studies/triangle_conditioning.py
python scripts/studies/device_conditioning.py --device cpu
python scripts/studies/device_conditioning.py --device cuda:0 \
  --output "$EVIDENCE/device-conditioning-cuda"
python scripts/studies/sdf_guard.py --device cpu
python scripts/studies/sdf_guard.py --device cuda:0 \
  --output "$EVIDENCE/sdf-guard-cuda"
python scripts/studies/sdf_distance_only.py
```

The atlas and broad survey independently generate their own current evidence:

```bash
python scripts/studies/boundary_atlas.py
# Preview: --radii 4 --grid 17 --cut 17 --dpi 150
python -m warpSPHBoundaries.validation.robustness --device cuda:0
# CPU-only mathematical survey: --skip-production; not production approval.
python -m warpSPHBoundaries.validation.sdf_corner_mechanism
```

Use `--report-only` where supported to redraw cached measurements.
`python -m warpSPHBoundaries.validation.robustness --replot` redraws the broad
survey; `python -m warpSPHBoundaries.validation.audit_report --refresh-stress-only
--output "$EVIDENCE"` refreshes report stress sections. No local web server is
required; reports include ordinary figure links and optional HTML companions.

Historical pre-fix results are archival evidence. Running current code produces
current behavior; recreating an old defect requires its recorded source state.
Do not present fresh measurements as reproductions of an archived pre-fix run.

## Regression and symbolic checks

```bash
python -m pytest tests/validation tests/scene/test_sdf_guard.py \
  tests/scene/test_complete_surface_moments.py -q
maple/run_edge.sh
maple -q maple/20_sdf_accuracy.mpl
maple -q maple/21_curvature_moments.mpl
```

CPU/GPU execution and Maple availability depend on the host. Keep those coverage
limits explicit. The archive records test logs and source provenance; test counts
and chronological progress belong there rather than in the API documentation.
