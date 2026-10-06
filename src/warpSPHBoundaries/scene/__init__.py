"""Layer B, the scene layer: bodies with surface / box / volume / implicit / SDF representations, the boundary operations in the warpSPH call shape, the wall operators of the delta+-SPH solver (cover
vector, cone area, tensile term, wall Laplacian) and the boundary PROVIDER a scheme consumes (`AnalyticBoundary`, `provider.py`).  See docs/scene-architecture.md.

Public surface (lazy: importing the package does not import warp / torch kernels):  Scene, Body, SurfaceRep, BoxRep, ImplicitRep, SdfRep, VolumeRep, DiskBody, HalfPlaneBody,
AnalyticBoundary, ParticleBoundary, BoundaryProvider, WallAggregate, WallOutput.  Layering (enforced by tests/test_package_layering.py): `edge` <- `scene` <- `sim`; `edge` and `scene` import `warpSPHCore` but never `warpSPH`.
"""
_EXPORTS = {
    "Scene": "scene", "Body": "scene", "SurfaceRep": "scene", "BoxRep": "scene", "ImplicitRep": "scene", "SdfRep": "scene", "VolumeRep": "scene", "BodyField": "scene", "sceneOperation": "scene",
    "DiskBody": "implicitBodies", "HalfPlaneBody": "implicitBodies", "TierPolicy": "implicitBodies",
    "AnalyticBoundary": "provider", "ParticleBoundary": "particles", "BoundaryProvider": "provider", "WallAggregate": "fused",
    "WallOutput": "fused", "FusedWall": "fused",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module("." + _EXPORTS[name], __name__), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
