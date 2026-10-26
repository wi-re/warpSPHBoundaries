"""Exact rectangular geometry through the shared analytic polygon family.

This is an explicit geometry adapter, not a new lookup or a different integral.
It retains the complete requested degree-nine/derivative-two mesh contract.
"""
import numpy as np
from .contract import baseline
from .geometry import Geometry
from .evaluate import Evaluator


class BoxEvaluator(Evaluator):
    """One or many disjoint/additive 2D rectangles, with optional rigid rotation.

    centres has shape (E,2), half_extents broadcasts to (E,2), and angles to E.
    Overlaps add; no implicit union is taken. Fields can use `self.geometry`
    with PolynomialField/FieldEvaluator just like other exact polygons.
    """
    def __init__(self,centres,half_extents,requests=None,*,angles=0.,kernel="w2",precision="float64",device="cpu",compact=True,method="analytic",nodes=16):
        centres=np.asarray(centres,dtype=float)
        if centres.shape==(0,):centres=centres.reshape(0,2)
        if centres.ndim==1:centres=centres.reshape(1,-1)
        if centres.shape!=(len(centres),2):
            raise ValueError("box centres must have shape (E,2)")
        count=len(centres)
        try:
            extents=np.asarray(half_extents,dtype=float)
            if count==0 and extents.shape==(0,):extents=extents.reshape(0,2)
            extents=np.broadcast_to(extents,(count,2)).copy()
            angle=np.broadcast_to(np.asarray(angles,dtype=float),(count,)).copy()
        except ValueError as exc:
            raise ValueError("incompatible box extents/angles") from exc
        if not all(np.isfinite(a).all() for a in (centres,extents,angle)) or np.any(extents<=0):
            raise ValueError("finite box inputs and positive half extents required")
        corners=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        elements=[]
        for c,e,a in zip(centres,extents,angle):
            ca,sa=np.cos(a),np.sin(a)
            rotation=np.array([[ca,-sa],[sa,ca]])
            elements.append((corners*e)@rotation.T+c)
        super().__init__(Geometry(elements,2),baseline(2) if requests is None else requests,
                         kernel=kernel,precision=precision,device=device,compact=compact,method=method,nodes=nodes)
        if precision=="float32" and method=="analytic":
            self.accuracy_status="experimental raw fp32 polygon recurrence; use the recorded box fixture failures"
