"""generates the warpSPH-style `wp_<fn>.py` modules (template of warpSPH `modules/surfaceDetection/wp_dilate.py` / `sample/wp_deltaShift.py`)."""
import os, textwrap
ROOT = "/home/lu26029/dev/curvatureBoundaries/src/warpSPHBoundaries/sim/modules"

TEMPLATE = '''"""@DOC@
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['@NAME@Warp']


@wp.func
def @NAME@_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32,

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.

    # Domain and kernel parameters
    domainState: domainData,
    kernelProperties: kernelState,

    # Operation specific parameters
    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore

    # Dummy value to allow allocation
    outputValue: Any, # type: ignore
@FUNC_PARAMS@
):
    # Initialize the output value
    out     = @OUT0@

    # Loop over neighbors
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
@BODY@

    return out


@wp.func
def @NAME@_Func_Adjacency(
    i : wp.int32, dim: wp.int32, lane: wp.int32, lanes: wp.int32,

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any,
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState,

    outputValue : Any, # type: ignore
@FUNC_PARAMS@
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out = @OUT0@
    for o in range(numOffsets):
        # grid traversal: lanes take whole cells round-robin (no-op for lanes == 1)
        if not useAdjacency and (o % lanes) != lane:
            continue
        beginIndex = wp.int32(0)
        numIndices = wp.int32(0)
        if useAdjacency:
            beginIndex = adjacencyState.neighborOffsets[i]
            numIndices = adjacencyState.numNeighbors[i]
        else:
            beginIndex, numIndices = checkOffset(
                i, queryState.positions, gridState.numCells, gridState.D,
                o, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
                domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell
            )
            if beginIndex < 0:
                continue

        beginIndex, numIndices = laneSlice(beginIndex, numIndices, lane, lanes, useAdjacency)
        partial = @NAME@_Func_i(
            i, dim,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi , correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,

            outputValue,
@FUNC_ARGS@
        )
        @ACCUM@
    return out


@wp.kernel
def @NAME@_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above
@KERNEL_PARAMS@
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = @OUTDT@) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = @NAME@_Func_Adjacency(
        i, domainState.dim, 0, 1,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed

        zero_like_warp(outputValues),
@FUNC_ARGS@
    )

@TILED@

def _@NAME@Dtype(ctx, extras):
    return @DTYPE_EXPR@


_@SPEC@ = OperatorSpec(
    kernel=@NAME@_Kernel,
@TILED_SPEC@    outputs=(OutputSpec(dtype=_@NAME@Dtype, shape=ShapeOf.QUERY),),
    extras=(
@EXTRA_SPECS@    ),
)


def @NAME@Warp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
@WRAP_PARAMS@
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[@NAME@]"):
        referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
        with record_function("warpSPH[@NAME@] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(_@SPEC@, ctx@LAUNCH_ARGS@)
'''

TILED = '''@wp.kernel
def @NAME@_KernelTiled(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above
@KERNEL_PARAMS@
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = @OUTDT@) # type: ignore
):
    # Multi-lane variant of @NAME@_Kernel (warpSPHCore autograd/lanes.py):
    # launched dim=[N, lanes]; each lane walks a slice of i's neighbours.

    i, lane = wp.tid()
    partial = @NAME@_Func_Adjacency(
        i, domainState.dim, lane, wp.block_dim(),
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        zero_like_warp(outputValues),
@FUNC_ARGS@
    )
    total = laneSum(partial)
    if lane == 0:
        outputValues[i] = total
'''

VEC = "vector(length=Any, dtype=scalar_t)"


def gen(task, fname, name, doc, extras, body, outkind, out0=None, accum="out += partial", tiled=True, spec=None):
    """extras: list of (name, kind, decl) with kind 'tensor' | 'scalar' | 'int'; decl = warp type string"""
    fp = "".join("\n    %s: %s, # type: ignore" % (n, d) for n, k, d in extras)
    kp = "".join("\n    %s: %s, # type: ignore" % (n, d) for n, k, d in extras)
    fa = "".join("\n            %s," % n for n, k, d in extras)
    ka = "".join("\n        %s," % n for n, k, d in extras)
    outdt = VEC if outkind == "vector" else "scalar_t"
    dtype_expr = "castTorchToWarpAsBuiltins(ctx.query.positions).dtype" if outkind == "vector" else "castTorchToWarpAsBuiltins(ctx.query.densities).dtype"
    out0 = out0 or "zero_like_warp(outputValue)"
    t = TEMPLATE
    tiled_txt = TILED if tiled else "# (no multi-lane variant: the reduction is a minimum, `laneSum` is a sum)"
    t = t.replace("@TILED@", tiled_txt)
    t = t.replace("@TILED_SPEC@", "    tiledKernel=@NAME@_KernelTiled,\n" if tiled else "")
    specs = "".join('        ExtraSpec("%s", ExtraKind.%s),\n' % (n, "TENSOR" if k == "tensor" else "SCALAR") for n, k, d in extras)
    wp_params = "".join("\n    %s: %s," % (n, "torch.Tensor" if k == "tensor" else ("int" if k == "int" else "float")) for n, k, d in extras)
    launch = "".join(", %s=%s" % (n, n if k == "tensor" else ("%s" % n if k == "int" else "scalar_t(%s)" % n)) for n, k, d in extras)
    for tok, val in (("@FUNC_PARAMS@", fp), ("@KERNEL_PARAMS@", kp), ("@FUNC_ARGS@", fa if False else None)):
        pass
    t = t.replace("@FUNC_PARAMS@", fp.rstrip(",") if False else fp)
    t = t.replace("@KERNEL_PARAMS@", kp)
    # FUNC_ARGS appear with different indentation in the three call sites: handle by the line before
    out = []
    for line in t.split("\n"):
        if line == "@FUNC_ARGS@":
            out.append(None)
        else:
            out.append(line)
    # indentation of FUNC_ARGS depends on the surrounding call (Func_i call: 12, kernel: 8)
    res, ctx_indent = [], 12
    for idx, line in enumerate(out):
        if line is None:
            prev = "\n".join(res[-6:])
            indent = 12 if "outputValue," in res[-1] else 8
            for n, k, d in extras:
                res.append(" " * indent + n + ",")
        else:
            res.append(line)
    t = "\n".join(res)
    t = t.replace("@BODY@", textwrap.indent(textwrap.dedent(body).strip("\n"), "        "))
    t = t.replace("@OUT0@", out0).replace("@ACCUM@", accum).replace("@OUTDT@", outdt).replace("@DTYPE_EXPR@", dtype_expr)
    t = t.replace("@EXTRA_SPECS@", specs).replace("@WRAP_PARAMS@", wp_params).replace("@LAUNCH_ARGS@", launch)
    t = t.replace("@DOC@", doc).replace("@NAME@", name).replace("@SPEC@", spec or name.upper())
    os.makedirs(os.path.join(ROOT, task), exist_ok=True)
    with open(os.path.join(ROOT, task, fname), "w") as f:
        f.write(t)


gen("surfaceDetection", "wp_barecascoCover.py", "computeBarecascoCover",
    "Barecasco cover vector C_i = sum_j unit(x_i - x_j) over the neighbours j != i within the support, the RAW sum of the first pass of warpSPH's `wp_barecasco.py` (which returns only the normalised vector and\nthe decision): a boundary representation adds its own cover contribution (an exact integral over the solid, or a wall-particle sum) to this partial sum before the direction is taken.  Neighbour count of the same\nset: `countNeighbors - 1` (warpSPH `modules/util`).",
    [], """
        if j == i:
            continue
        if r_ij > hij:
            continue
        out += x_ij / wp.max(r_ij, scalar_t(1.0e-30))
""", "vector")

gen("surfaceDetection", "wp_barecascoCone.py", "computeBarecascoConeCount",
    "Second pass of the Barecasco detector as a PARTIAL SUM: the number of neighbours j != i within the support whose direction x_j - x_i lies within `halfAngle` of the axis `coverAxes[i]`\n(acos(-unit(x_i - x_j) . c_i) <= halfAngle; `c_i` the normalised cover vector).  warpSPH's `wp_barecasco.py` takes the decision inside the kernel; a boundary representation adds the area of its solid in the same\nwedge to this count before the threshold test.",
    [("coverAxes", "tensor", "wp.array(dtype = " + VEC + ")"), ("halfAngle", "scalar", "scalar_t")], """
        if j == i:
            continue
        if r_ij > hij:
            continue
        u = x_ij / wp.max(r_ij, scalar_t(1.0e-30))
        cosang = -wp.dot(u, coverAxes[i])
        if wp.acos(wp.clamp(cosang, scalar_t(-1.0), scalar_t(1.0))) <= halfAngle:
            out += scalar_t(1.0)
""", "scalar")

gen("shifting", "wp_deltaShiftRaw.py", "computeDeltaShiftRaw",
    "Raw delta+ shift sum  sum_{j != i, r <= h} 0.5 m_j / (rho_i + rho_j) [1 + R (W_ij / W0)^n] grad_i W_ij  (Sun et al. 2017 Eq. (7) without the Mach / CFL scaling) with the tensile reference value `W0` GIVEN\nexplicitly (the kernel at the particle spacing).  warpSPH's `computeDeltaShiftWarp` (sample/wp_deltaShift.py) is the same sum with W0 evaluated inside from the mass (`q = (m/rho0)^(1/dim) / kernelScale / h`:\nthe kernel at dx / kernelScale, which differs from the kernel at dx by (W(dx) / W(dx / kernelScale))^n in the tensile term, 3.7 % of the sum for Wendland C2 at h = 4 dx; docs/audit-warpsph-boundary-hooks.md §6).\nA boundary representation adds the wall part of the same sum before the surface treatment.",
    [("R", "scalar", "scalar_t"), ("n", "int", "wp.int32"), ("W0", "scalar", "scalar_t")], """
        if j == i:
            continue
        if r_ij > hij:
            continue
        w_ij = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi
        )
        gradw_ij = computeKernelGradientCRK(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        k = w_ij / W0
        out += scalar_t(0.5) * mj / (rhoi + rhoj) * (scalar_t(1.0) + R * wp.pow(k, scalar_t(n))) * gradw_ij
""", "vector")

gen("shifting", "wp_minNeighbourNormalDot.py", "computeMinNeighbourNormalDot",
    "Smallest dot product n_i . n_j of the unit surface normals over the neighbours j != i within the support, restricted to particles in the mask (`surfaceMask[i] != 0` and `surfaceMask[j] != 0`); a large number where\nthere is none.  The curvature test of the delta+ surface shift (Sun et al. 2019: a shift towards the surface is only tangential where a neighbour's normal differs by more than the threshold angle).",
    [("surfaceMask", "tensor", "wp.array(dtype = wp.int32)"), ("surfaceNormals", "tensor", "wp.array(dtype = " + VEC + ")")], """
        if surfaceMask[i] == 0:
            continue
        if j == i or surfaceMask[j] == 0:
            continue
        if r_ij > hij:
            continue
        out = wp.min(out, wp.dot(surfaceNormals[i], surfaceNormals[j]))
""", "scalar", out0="outputValue + scalar_t(1.0e30)", accum="out = wp.min(out, partial)", tiled=False)
