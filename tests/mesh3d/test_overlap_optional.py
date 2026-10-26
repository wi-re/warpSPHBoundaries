"""Independent authors' implementation; optional, never a runtime dependency."""
import mpmath as mp
import numpy as np
import pytest

from warpSPHBoundaries.mesh3d import Tetrahedron

overlap = pytest.importorskip("overlap")


@pytest.mark.parametrize("query,radius", [((0, 0, 0), .2), ((.4, 0, 0), .1),
                                        ((.2, .2, .2), .3), ((.1, -.2, .3), .5),
                                        ((.2, .2, .2), 2.), ((-2, 0, 0), .1)])
def test_constant_kernel_overlap(query, radius):
    vertices = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
    with mp.workdps(50):
        t = Tetrahedron(vertices, query)
        got = t.block_moment((0, 0, 0), 0, mp.mpf(radius))
        ref = overlap.overlap_volume(overlap.Sphere(query, radius), overlap.Tetrahedron(vertices))
        assert abs(float(got)-ref) < 5e-15
