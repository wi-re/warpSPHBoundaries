# Reusable Warp boundary evaluators

Strengthening an existing representation includes replacing small query-time
Torch dispatches with composable Warp functions, plus verified gradients and
adjoints. Each migrated component has a folder containing a host wrapper and
`wp_func.py`. Host wrappers prepare static data and launch batches; device
functions perform per-query or per-interaction arithmetic without allocation
or host dispatch. Batch and fused consumers must use the same device function.

The first extracted component is the **existing order-two curvature model**:

```
src/warpSPHBoundaries/scene/curvature/
    __init__.py
    curvature.py  # table packing, direct batches, Torch first-order VJP bridge
    wp_func.py    # table struct, spline primitives, moment-function factory
```

This extraction retains the model and coefficient splines. The guarded Scene
SDF kernel calls this same function. Its geometry sampling and guard policy
remain in `scene/sdf_warp.py` and are a separate forward-only contract.

## Direct batches

```python
from warpSPHBoundaries.scene.implicitBodies import Tier3
from warpSPHBoundaries.scene.curvature import evaluate

table = Tier3('w4', device)
# Tensors: distance [N], normal [N,2], curvature [N], variation [N].
values = evaluate(distance, normal, curvature, variation, table)
# First-order Torch autograd propagates to those four jet inputs.
```

Inputs are dimensionless `d/h`, unit normal, `h*kappa`, and
`h**2 * d(kappa)/ds`, with tangent `(-normal_y, normal_x)`. The canonical output
order is fixed across forward and derivative checks:

| Columns | Quantity | Multiply by for physical moments |
|---|---|---|
| 0 | m00 | 1 |
| 1, 2 | m10, m01 | h |
| 3, 4, 5 | m20, m11, m02 | h² |
| 6, 7 | Gx, Gy | 1/h |
| 8–11 | g1xx, g1xy, g1yx, g1yy | 1 |

`channels` follows Scene's legacy IDs, which differ from column order:
`(0,1,2,3,4,5,6,7,8,9,10,11)` selects canonical columns
`(0,1,2,6,7,8,9,10,11,3,4,5)`. Unrequested columns and their derivatives are
zero. Computation uses FP64; no float32 accuracy claim follows from this port.

## Calling from an interaction kernel

Prepare the frozen table once and specialize the device function outside the
consumer kernel. This also exposes a direct many-particles/one-plane path:

```python
import warp as wp
from warpSPHBoundaries.scene.curvature import (
    CurvatureTable, moment_function, prepare_table,
)

moments = moment_function(channels=(0, 1, 2))
packed = prepare_table(table)  # outside the kernel/capture/timed loop

@wp.kernel
def plane_mass(positions: wp.array(dtype=wp.vec2d),
               supports: wp.array(dtype=wp.float64),
               normal: wp.vec2d, offset: wp.float64,
               coefficients: CurvatureTable,
               mass: wp.array(dtype=wp.float64)):
    i = wp.tid()
    q = (wp.dot(positions[i], normal)-offset)/supports[i]
    value = moments(q, normal, wp.float64(0.), wp.float64(0.), coefficients)
    mass[i] = value[0]
```

Consumers may instead accumulate boundary interactions directly, avoiding
intermediate moment tensors. The table coefficients are frozen. Keep the
`Tier3` owner alive, reuse the packed struct, and rebuild it if coefficient
storage changes. Physical support scaling and geometric parameterization
belong in the consumer and are differentiated there.

## Derivative contract and limitations

- The smooth model supports Warp reverse-mode Tape and a first-order Torch
  VJP bridge for distance, normal components, curvature and curvature variation.
  Unit normals are a precondition; differentiating their components treats them
  as independent algebraic inputs. Compose an angle/normal parameterization
  when geometric derivatives are needed.
- Integer spline-segment selection has an explicit zero adjoint. Spline
  evaluation uses explicit first/second Hermite derivatives and spline
  integration uses the fundamental theorem of calculus, with frozen
  coefficients. These differentiate the executed piece; this does not assert
  differentiability at knots, the sign switch, or empty/full support switches.
- Full/empty branches are constant in unit-support coordinates. Physical
  moment scaling can still have a nonzero support derivative.
- The output `G` is an integral of the kernel gradient, not a promise that it
  equals differentiation of the entire sampled/guarded approximation. Model
  truncation and jet reconstruction must remain explicit in that comparison.
- The API does not validate geometric admissibility, certify integral error,
  or select fallback. It assumes finite valid jets for partial support. Use
  the guarded Scene path when its selection/fallback policy is required.
- Derivatives of raw SDF grid values, preprocessing stencils, guard decisions,
  polygon fallback, full Scene assembly and higher-order autograd are not
  established by these local-model checks. Geometry-parameter adjoints for
  each representation remain explicit follow-up work.

## Verification and reproduction

`tests/scene/test_curvature_warp.py` compares every requested output and its
VJP against the retained Torch model for four kernels and sparse channel
requests. It checks directional finite differences, repeated backward,
empty/full support and empty batches. An independently defined Warp consumer
composes a plane, support scaling and a weighted moment loss and checks
particle-position, support, plane-offset and plane-angle adjoints.
`tests/scene/test_sdf_warp.py` checks the integrated guarded path, selection
parity and graph replay. Run in an environment with Warp, Torch and Core:

```sh
python -m pytest tests/scene/test_curvature_warp.py tests/scene/test_sdf_warp.py -q
```

CPU and CUDA must be reported separately, with any unavailable device stated.
Derivative agreement validates the executed local model; its geometric error
is unchanged. Timing evidence must distinguish device-function cost, the
direct batch wrapper, backward recomputation, and complete Scene evaluation.

This layout and verification contract should be applied incrementally to the
existing surface/triangle, half-plane, disk/box, volume and sampled-SDF paths.
Forward parity alone is insufficient to mark an entire representation's
adjoints complete.
