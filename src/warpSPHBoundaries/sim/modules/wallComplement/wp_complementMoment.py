"""Discrete fluid moment of the Morris pair weight in the frame of a wall (the complement closure of docs/plan-next-steps.md "first row"): sum_{j != i, r <= h} mu_ij K_ij s_j^k with K = (x_ij . grad W_ij) / (r^2 + eta2 h_i^2),
mu_ij = V_j (rho_i + rho_j) / (2 rho_i), s_j = (x_j - x_i) . n_i + d_i the distance coordinate of the neighbour in the frame of particle i's wall (`wallNormals`, `wallDistances`), k = `power` (0, 1, 2).
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeComplementMomentWarp']


@wp.func
def computeComplementMoment_Func_i(
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

    wallNormals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    wallDistances: wp.array(dtype = scalar_t), # type: ignore
    eta2: scalar_t, # type: ignore
    power: wp.int32, # type: ignore
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
        gradw_ij = computeKernelGradientCRK(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        K = wp.dot(x_ij, gradw_ij) / (r_ij * r_ij + eta2 * hi * hi)
        mu = mj / rhoj * (rhoi + rhoj) / (scalar_t(2.0) * rhoi)
        s = wallDistances[i] - wp.dot(x_ij, wallNormals[i])
        sk = scalar_t(1.0)
        for p in range(power):
            sk = sk * s
        out += mu * K * sk

    return out


@wp.func
def computeComplementMoment_Func_Adjacency(
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

    wallNormals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    wallDistances: wp.array(dtype = scalar_t), # type: ignore
    eta2: scalar_t, # type: ignore
    power: wp.int32, # type: ignore
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
        partial = computeComplementMoment_Func_i(
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
            wallNormals,
            wallDistances,
            eta2,
            power,
        )
        out += partial
    return out


@wp.kernel
def computeComplementMoment_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above

    wallNormals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    wallDistances: wp.array(dtype = scalar_t), # type: ignore
    eta2: scalar_t, # type: ignore
    power: wp.int32, # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeComplementMoment_Func_Adjacency(
        i, domainState.dim, 0, 1,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed

        zero_like_warp(outputValues),
        wallNormals,
        wallDistances,
        eta2,
        power,
    )

@wp.kernel
def computeComplementMoment_KernelTiled(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above

    wallNormals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    wallDistances: wp.array(dtype = scalar_t), # type: ignore
    eta2: scalar_t, # type: ignore
    power: wp.int32, # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):
    # Multi-lane variant of computeComplementMoment_Kernel (warpSPHCore autograd/lanes.py):
    # launched dim=[N, lanes]; each lane walks a slice of i's neighbours.

    i, lane = wp.tid()
    partial = computeComplementMoment_Func_Adjacency(
        i, domainState.dim, lane, wp.block_dim(),
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        zero_like_warp(outputValues),
        wallNormals,
        wallDistances,
        eta2,
        power,
    )
    total = laneSum(partial)
    if lane == 0:
        outputValues[i] = total


def _computeComplementMomentDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_COMPUTECOMPLEMENTMOMENT = OperatorSpec(
    kernel=computeComplementMoment_Kernel,
    tiledKernel=computeComplementMoment_KernelTiled,
    outputs=(OutputSpec(dtype=_computeComplementMomentDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("wallNormals", ExtraKind.TENSOR),
        ExtraSpec("wallDistances", ExtraKind.TENSOR),
        ExtraSpec("eta2", ExtraKind.SCALAR),
        ExtraSpec("power", ExtraKind.SCALAR),
    ),
)


def computeComplementMomentWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    wallNormals: torch.Tensor,
    wallDistances: torch.Tensor,
    eta2: float,
    power: int,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[computeComplementMoment]"):
        referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
        with record_function("warpSPH[computeComplementMoment] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(_COMPUTECOMPLEMENTMOMENT, ctx, wallNormals=wallNormals, wallDistances=wallDistances, eta2=scalar_t(eta2), power=power)
