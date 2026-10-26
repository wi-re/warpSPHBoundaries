"""Analytic, arbitrary-precision 3D mesh reference (not a Scene/GPU backend).

Use :class:`Tetrahedron` for volume elements, :class:`ClosedSurface` for
oriented watertight triangle meshes, and :class:`Triangle` for surface area
integrals. Coordinates and support radii are physical; y = source - query.
"""

from .reference import ClosedSurface, TetrahedralMesh, Tetrahedron, Triangle, kernel_profile

__all__ = ["ClosedSurface", "TetrahedralMesh", "Tetrahedron", "Triangle", "kernel_profile"]
