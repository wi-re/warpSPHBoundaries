"""Explicit weighted solid-sphere shell quadrature with two-order control.

Supports Cartesian moments through degree nine and kernel derivatives through
order two. Geometry is a sphere; no curvature model or polygon approximation
is substituted. Analytic radial baseline evaluation remains a separate API.
"""
import numpy as np
import warp as wp
from .contract import baseline
from .planar import PlanarBatch
from .planar_shell import setup, functions as shell_functions

MAX_BATCH_BYTES = 2**30


def functions(precision='float64', nodes=24):
    """Composable normalized sphere functions, with caller-owned scratch.

    `element(distance/h, radius/h, normal, plan, indices, facets, work, row)`
    uses outward normal from centre to query. At the centre any unit axis works.
    `output` is normalized; multiply once by h**Request.scale for physical units.
    """
    return shell_functions(precision,nodes,geometry='sphere')


class SphereShellEvaluator:
    """Numerical weighted sphere provider in ordinary fp32/fp64.

    Inputs are signed outward query-to-sphere distances, unit centre-to-query
    normals, support radii and sphere radii. Default radial order is 24. Prepared
    rules compare to order+8 and may promote up to 64; unresolved inputs reject.
    Finite convergence checks do not certify global accuracy or positivity.
    """
    dimension = 3

    def __init__(self,requests=None,*,kernel='w2',precision='float64',device='cpu',nodes=24):
        self.requests=baseline(3) if requests is None else tuple(requests)
        self.kernel,self.precision,self.device=kernel,precision,wp.get_device(device)
        setup(self,nodes,geometry='sphere')
        self.nodes=nodes
        self.method='sphere-shell-quadrature'
        self.accuracy_status='explicit two-order numerical rule; independent acceptance is finite'
        self._orders={}

    def _at_order(self,order):
        if order==self.nodes:return self
        if order not in self._orders:
            self._orders[order]=SphereShellEvaluator(self.requests,kernel=self.kernel,
                precision=self.precision,device=self.device,nodes=order)
        return self._orders[order]

    def prepare(self,distances,normals,supports,*,radii):
        d=np.atleast_1d(np.asarray(distances,dtype=float))
        if d.ndim!=1:raise ValueError('distances must be scalar or vector')
        count=len(d)
        try:
            n=np.broadcast_to(np.asarray(normals,dtype=float),(count,3)).copy()
            h=np.broadcast_to(np.asarray(supports,dtype=float),(count,)).copy()
            r=np.broadcast_to(np.asarray(radii,dtype=float),(count,)).copy()
        except ValueError as error:raise ValueError('incompatible sphere batch shapes') from error
        if not all(np.isfinite(a).all() for a in (d,n,h,r)) or np.any(h<=0) or np.any(r<=0) or np.any(d < -r):
            raise ValueError('finite sphere inputs, positive support/radius and distance >= -radius required')
        if not np.allclose(np.linalg.norm(n,axis=1),1,rtol=0,atol=1e-12):
            raise ValueError('normals must be unit vectors')
        dtype=np.float32 if self.precision=='float32' else np.float64
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            dd=np.asarray(d.astype(np.longdouble)/h.astype(np.longdouble),dtype=dtype)
            rr=np.asarray(r.astype(np.longdouble)/h.astype(np.longdouble),dtype=dtype)
            hh=h.astype(dtype)
            factors=np.array([hh**req.scale for req in self.requests])
            intermediate=np.array([hh**abs(req.scale) for req in self.requests])
            valid=(all(np.isfinite(a).all() for a in (dd,rr,hh,factors,intermediate)) and
                   np.all(rr>0) and np.all(hh>0) and np.all(factors!=0) and np.all(intermediate!=0) and
                   np.isfinite(rr+rr+abs(dd)).all())
        if not valid:raise ValueError('sphere geometry/support scaling exceeds finite precision storage')
        budget=1e-5 if self.precision=='float32' else 1e-14
        order=self.nodes;promotions=0
        while True:
            comparison=min(64,order+8)
            owner=self._at_order(order);other=self._at_order(comparison)
            size=np.dtype(dtype).itemsize
            estimate=lambda e:count*size*(9*e.api.face_count+3*len(self.requests)+6)
            if max(estimate(owner),estimate(other))>MAX_BATCH_BYTES:
                raise ValueError('sphere shell rule storage exceeds 1 GiB batch budget; split queries')
            b=PlanarBatch(owner,dd,rr,n,h)
            if not count or not self.requests:
                b.quadrature=dict(requested_nodes=self.nodes,nodes=order,comparison_nodes=None,
                    worst_mixed_difference=0.,budget=budget,promotions=0,samples=0)
                return b
            if comparison==order:raise ValueError('sphere shell needs a higher comparison order below 64')
            reference=PlanarBatch(other,dd,rr,n,h)
            b.launch();reference.launch()
            x=b.work.numpy();y=reference.work.numpy()
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('sphere shell produced nonfinite normalized values')
            if not np.isfinite(b.values.numpy()).all():
                raise ValueError('sphere shell physical output exceeds finite precision storage')
            error=float(np.max(abs(x-y)/(1+abs(y))))
            del reference,x,y
            if error<=budget:
                b.quadrature=dict(requested_nodes=self.nodes,nodes=order,comparison_nodes=comparison,
                    worst_mixed_difference=error,budget=budget,promotions=promotions,
                    samples=count*owner.api.face_count,geometry_precision=self.precision,
                    control='two-order normalized moment comparison; not a certified error bound')
                return b
            del b
            order=comparison;promotions+=1

    def evaluate(self,distances,normals,supports,*,radii):
        b=self.prepare(distances,normals,supports,radii=radii)
        b.launch()
        return b.numpy()
