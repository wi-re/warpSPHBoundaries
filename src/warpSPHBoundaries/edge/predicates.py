"""Orientation arithmetic shared by indicators and edge integral jumps.

The Torch indicator rounds each product before subtracting. CUDA FMA contraction
of a*b-c*d can otherwise make even p=-q appear noncollinear. Explicit CUDA
round-to-nearest intrinsics and volatile CPU products preserve those roundings
without disabling FMA for the surrounding integral. This is a consistent floating predicate, not an exact
adaptive orientation test. Callers must use identical relative coordinates and
support normalization.
"""
import warp as wp


_ADJOINT = """
adj_ax += by * adj_ret;
adj_ay -= bx * adj_ret;
adj_bx -= ay * adj_ret;
adj_by += ax * adj_ret;
"""


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __dsub_rn(__dmul_rn(ax, by), __dmul_rn(ay, bx));
#else
volatile double left = ax * by;
volatile double right = ay * bx;
return left - right;
#endif
""", adj_snippet=_ADJOINT)
def cross2_f64(ax: wp.float64, ay: wp.float64, bx: wp.float64, by: wp.float64) -> wp.float64:
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __fsub_rn(__fmul_rn(ax, by), __fmul_rn(ay, bx));
#else
volatile float left = ax * by;
volatile float right = ay * bx;
return left - right;
#endif
""", adj_snippet=_ADJOINT)
def cross2_f32(ax: wp.float32, ay: wp.float32, bx: wp.float32, by: wp.float32) -> wp.float32:
    ...
