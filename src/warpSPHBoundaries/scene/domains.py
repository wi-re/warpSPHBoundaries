"""Scenes of a closed box of fluid: the domain walls as one body in several equivalent representations (surface loop, box primitive, omniSPH-style volume slabs, sampled SDF, half planes).

Moved here from the retired `sim.dfsph2d` (ANALYTIC_BOUNDARIES_PLAN E7); the representation-agreement tests of the scene layer use it as a fixture.
"""
from typing import Optional

import numpy as np
import torch

from .implicitBodies import HalfPlaneBody
from .scene import Body, BoxRep, ImplicitRep, Scene, SdfRep, SurfaceRep, VolumeRep

F64 = torch.float64


def domain_scene(kind: str, lo, hi, h: float, device, thickness: Optional[float] = None, volumeMode: str = "moments"):
    """closed box of fluid [lo, hi] (the INNER wall faces) as one body (`volume`: omniSPH's thick triangle slabs, `surface`: one clockwise loop in an infinite solid,
    `box`: the same loop as a `BoxRep`, `sdf`: sampled box SDF with the loop as fallback near the corners, `halfplanes`: four half planes (negative control: double counts the corner regions))."""
    (x0, y0), (x1, y1) = lo, hi
    t = thickness if thickness is not None else 2.5 * h
    if kind == "surface":
        return Scene([Body(reps=[SurfaceRep.box((x0, y0), (x1, y1), solid="outside")])], device)
    if kind == "box":                                 # the same domain as a BoxRep: corner tables, no edge pairs (docs/box-domain-primitive.md)
        return Scene([Body(reps=[BoxRep((x0, y0), (x1, y1), solid="outside")])], device)
    if kind == "volume":
        V = np.array([[x0 - t, y0 - t], [x0, y0 - t], [x1, y0 - t], [x1 + t, y0 - t],
                      [x0 - t, y0], [x0, y0], [x1, y0], [x1 + t, y0],
                      [x0 - t, y1], [x0, y1], [x1, y1], [x1 + t, y1],
                      [x0 - t, y1 + t], [x0, y1 + t], [x1, y1 + t], [x1 + t, y1 + t]], dtype=float)
        E = []
        for r in range(3):
            for c in range(4 - 1):
                if (r, c) == (1, 1):                 # the fluid cell
                    continue
                a, b, cc, d = 4 * r + c, 4 * r + c + 1, 4 * (r + 1) + c, 4 * (r + 1) + c + 1
                E += [[a, b, d], [a, d, cc]]
        return Scene([Body(reps=[VolumeRep(V, np.array(E))])], device, volumeMode=volumeMode)
    if kind == "sdf":
        c = torch.tensor([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=F64)
        hw = torch.tensor([0.5 * (x1 - x0), 0.5 * (y1 - y0)], dtype=F64)

        def boxsdf(p):                              # positive inside the fluid box
            q = (p - c).abs() - hw
            return -(q.clamp(min=0).norm(dim=1) + q.max(dim=1).values.clamp(max=0))
        m = 2 * h
        sdf = SdfRep.fromFunction(boxsdf, (x0 - m, y0 - m), (x1 + m, y1 + m), h / 16, fallback=SurfaceRep.box((x0, y0), (x1, y1), solid="outside"))
        return Scene([Body(reps=[sdf])], device)
    if kind == "halfplanes":
        planes = [((x0, 0.0), (1.0, 0.0)), ((x1, 0.0), (-1.0, 0.0)), ((0.0, y0), (0.0, 1.0)), ((0.0, y1), (0.0, -1.0))]
        return Scene([Body(bodyId=i, reps=[ImplicitRep(HalfPlaneBody(p, n))]) for i, (p, n) in enumerate(planes)], device)
    raise ValueError(kind)
