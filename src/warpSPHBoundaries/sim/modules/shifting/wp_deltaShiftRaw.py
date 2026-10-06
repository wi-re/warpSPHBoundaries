"""Raw delta+ shift sum  sum_{j != i, r <= h} 0.5 m_j / (rho_i + rho_j) [1 + R (W_ij / W0)^n] grad_i W_ij  (Sun et al. 2017 Eq. (7) without the Mach / CFL scaling) with the tensile reference value `W0` GIVEN
explicitly (the kernel at the particle spacing).  warpSPH's `computeDeltaShiftWarp` (sample/wp_deltaShift.py) is the same sum with W0 evaluated inside from the mass (`q = (m/rho0)^(1/dim) / kernelScale / h`:
the kernel at dx / kernelScale, which differs from the kernel at dx by (W(dx) / W(dx / kernelScale))^n in the tensile term, 3.7 % of the sum for Wendland C2 at h = 4 dx; docs/audit-warpsph-boundary-hooks.md §6).
A boundary representation adds the wall part of the same sum before the surface treatment.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeDeltaShiftRawWarp']


@wp.func
def computeDeltaShiftRaw_Func_i(
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

    R: scalar_t, # type: ignore
    n: wp.int32, # type: ignore
    W0: scalar_t, # type: ignore
):
    # Initialize the output value
    out     = zero_like_warp(outputValue)

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

    return out


@wp.func
def computeDeltaShiftRaw_Func_Adjacency(
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

    R: scalar_t, # type: ignore
    n: wp.int32, # type: ignore
    W0: scalar_t, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out = zero_like_warp(outputValue)
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
        partial = computeDeltaShiftRaw_Func_i(
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
            R,
            n,
            W0,
        )
        out += partial
    return out


@wp.kernel
def computeDeltaShiftRaw_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above

    R: scalar_t, # type: ignore
    n: wp.int32, # type: ignore
    W0: scalar_t, # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = vector(length=Any, dtype=scalar_t)) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeDeltaShiftRaw_Func_Adjacency(
        i, domainState.dim, 0, 1,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed

        zero_like_warp(outputValues),
        R,
        n,
        W0,
    )

@wp.kernel
def computeDeltaShiftRaw_KernelTiled(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above

    R: scalar_t, # type: ignore
    n: wp.int32, # type: ignore
    W0: scalar_t, # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = vector(length=Any, dtype=scalar_t)) # type: ignore
):
    # Multi-lane variant of computeDeltaShiftRaw_Kernel (warpSPHCore autograd/lanes.py):
    # launched dim=[N, lanes]; each lane walks a slice of i's neighbours.

    i, lane = wp.tid()
    partial = computeDeltaShiftRaw_Func_Adjacency(
        i, domainState.dim, lane, wp.block_dim(),
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        zero_like_warp(outputValues),
        R,
        n,
        W0,
    )
    total = laneSum(partial)
    if lane == 0:
        outputValues[i] = total


def _computeDeltaShiftRawDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_COMPUTEDELTASHIFTRAW = OperatorSpec(
    kernel=computeDeltaShiftRaw_Kernel,
    tiledKernel=computeDeltaShiftRaw_KernelTiled,
    outputs=(OutputSpec(dtype=_computeDeltaShiftRawDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("R", ExtraKind.SCALAR),
        ExtraSpec("n", ExtraKind.SCALAR),
        ExtraSpec("W0", ExtraKind.SCALAR),
    ),
)


def computeDeltaShiftRawWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    R: float,
    n: int,
    W0: float,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[computeDeltaShiftRaw]"):
        referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
        with record_function("warpSPH[computeDeltaShiftRaw] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(_COMPUTEDELTASHIFTRAW, ctx, R=scalar_t(R), n=n, W0=scalar_t(W0))
