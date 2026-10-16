"""Conditioned triangle area integration on CPU/CUDA, without host callbacks.

Legacy channel order is retained and extended: lam,m1x,m1y,Gx,Gy,
g1xx,g1xy,g1yx,g1yy,m20,m11,m02, all in units of h. Static geometry is
prepared once; positions, supports, validity, estimates and status stay on device.
Status 0 means analytic route, 1 outer quadrature converged, -1 exhausted budget.
Refinement estimates are empirical, not certified bounds.
"""
from fractions import Fraction
from functools import lru_cache

import numpy as np
import torch
import warp as wp

from .kernels import kernel as get_kernel
from .np_fem import kernel_ubasis
from .precision import sync

wf = wp.float64
v2 = wp.vec2d
v12 = wp.types.vector(length=12, dtype=wf)
v32 = wp.types.vector(length=32, dtype=wf)
v64 = wp.types.vector(length=64, dtype=wf)


@wp.func
def _add_cut(cuts: v32, n: int, t: wf):
    if t > wf(0.) and t < wf(1.) and n < 32:
        cuts[n] = t
        n += 1
    return cuts, n


@wp.func
def _cuts(cuts: v32, n: int, a: v2, b: v2, radii: wp.array(dtype=wf)):
    bb = wp.dot(b,b)
    if bb > wf(0.):
        t = -wp.dot(a,b)/bb
        cuts,n = _add_cut(cuts,n,t)
        closest = a+t*b
        rr = wp.dot(closest,closest)
        for j in range(radii.shape[0]):
            delta = (radii[j]*radii[j]-rr)/bb
            if delta >= wf(0.):
                root = wp.sqrt(delta)
                cuts,n = _add_cut(cuts,n,t-root)
                cuts,n = _add_cut(cuts,n,t+root)
    return cuts,n


@wp.func
def _sort(cuts: v32, n: int):
    for i in range(1,n):
        key = cuts[i]
        j = i-1
        while j >= 0:
            if cuts[j] <= key:
                break
            cuts[j+1] = cuts[j]
            j -= 1
        cuts[j+1] = key
    return cuts


@wp.func
def _line(u: wf, a: v2, b: v2, c: v2, scales: v12,
          radii: wp.array(dtype=wf), ubR: wp.array(dtype=wf), ubK: wp.array(dtype=int), ubB: wp.array(dtype=wf),
          origin: wp.array(dtype=wf), origin_hi: wf, gx: wp.array(dtype=wf), gw: wp.array(dtype=wf), offset: int, count: int):
    start = a+u*b
    direction = (wf(1.)-u)*c
    cuts = v32(wf(0.))
    cuts[1] = wf(1.)
    cuts,n = _cuts(cuts,2,start,direction,radii)
    cuts = _sort(cuts,n)
    out = v12(wf(0.))
    for part in range(n-1):
        lo,hi = cuts[part],cuts[part+1]
        if hi > lo:
            for j in range(count):
                t = lo+(hi-lo)*gx[offset+j]
                y = start+t*direction
                r = wp.length(y)
                W = wf(0.)
                dw = wf(0.)
                for k in range(ubR.shape[0]):
                    if r < ubR[k]:
                        z = ubR[k]-r
                        W += ubB[k]*wp.pow(z, wf(ubK[k]))
                        if ubK[k] > 0:
                            dw -= wf(ubK[k])*ubB[k]*wp.pow(z,wf(ubK[k]-1))
                radial = -dw/wp.max(r,wf(1e-300))
                if origin[1] == wf(0.) and r < origin_hi:
                    radial = wf(0.)
                    for kk in range(origin.shape[0]-2):
                        k = origin.shape[0]-1-kk
                        radial = radial*r-wf(k)*origin[k]
                Gx,Gy = radial*y[0],radial*y[1]
                values = v12(W,y[0]*W,y[1]*W,Gx,Gy,y[0]*Gx,y[0]*Gy,y[1]*Gx,y[1]*Gy,
                             y[0]*y[0]*W,y[0]*y[1]*W,y[1]*y[1]*W)
                weight = (hi-lo)*gw[offset+j]*(wf(1.)-u)/wf(3.1415926535897932384626433832795)
                for k in range(12):
                    out[k] += weight*values[k]/scales[k]
    return out


@wp.func
def _panel(lo: wf, hi: wf, a: v2, b: v2, c: v2, scales: v12,
           radii: wp.array(dtype=wf), ubR: wp.array(dtype=wf), ubK: wp.array(dtype=int), ubB: wp.array(dtype=wf),
           origin: wp.array(dtype=wf), origin_hi: wf, gx: wp.array(dtype=wf), gw: wp.array(dtype=wf), fine: int):
    out = v12(wf(0.))
    offset,count = int(0),int(8)
    inner_offset,inner_count = int(24),int(32)
    if fine != 0:
        offset,count = 8,16
    if fine < 0:
        inner_offset,inner_count = 8,16
    for j in range(count):
        u = lo+(hi-lo)*gx[offset+j]
        value = _line(u,a,b,c,scales,radii,ubR,ubK,ubB,origin,origin_hi,gx,gw,inner_offset,inner_count)
        for k in range(12):
            out[k] += (hi-lo)*gw[offset+j]*value[k]
    return out


@wp.kernel
def _triangle_kernel(pos: wp.array(dtype=v2), sup: wp.array(dtype=wf), valid: wp.array(dtype=int),
                     vertices: wp.array(dtype=v2), order: wp.array(dtype=int), area2: wf, sign: wf, background: wf,
                     radii: wp.array(dtype=wf), ubR: wp.array(dtype=wf), ubK: wp.array(dtype=int), ubB: wp.array(dtype=wf),
                     origin: wp.array(dtype=wf), origin_hi: wf, gx: wp.array(dtype=wf), gw: wp.array(dtype=wf),
                     plane: wp.array(dtype=wf), tolerance: wf, max_panels: int,
                     out: wp.array2d(dtype=wf), estimate: wp.array2d(dtype=wf), status: wp.array(dtype=int)):
    row = wp.tid()
    if valid[row] == 0:
        return
    h = sup[row]
    v0,v1,v2_ = vertices[order[0]],vertices[order[1]],vertices[order[2]]
    e0,e1 = v1-v0,v2_-v0
    longest2 = wp.max(wp.dot(e0,e0),wp.max(wp.dot(e1,e1),wp.dot(e1-e0,e1-e0)))
    if longest2 > wf(.02)*wf(.02)*h*h and area2 > wf(.01)*longest2:
        return
    status[row] = 1
    a,b,c = (v0-pos[row])/h,e0/h,e1/h
    sx = wp.max(wp.abs(a[0]),wp.max(wp.abs(a[0]+b[0]),wp.abs(a[0]+c[0])))
    sy = wp.max(wp.abs(a[1]),wp.max(wp.abs(a[1]+b[1]),wp.abs(a[1]+c[1])))
    sx,sy = wp.max(sx,wf(1e-150)),wp.max(sy,wf(1e-150))
    scales = v12(wf(1.),sx,sy,sx,sy,sx*sx,sx*sy,sy*sx,sy*sy,sx*sx,sx*sy,sy*sy)
    cuts = v32(wf(0.))
    cuts[1] = wf(1.)
    cuts,n = _cuts(cuts,2,a,b,radii)
    cuts,n = _cuts(cuts,n,a+c,b-c,radii)
    bc = b[0]*c[1]-b[1]*c[0]
    ac = a[0]*c[1]-a[1]*c[0]
    if bc != wf(0.):
        cuts,n = _add_cut(cuts,n,-ac/bc)
        for j in range(radii.shape[0]):
            for s in range(2):
                sg = wf(2*s-1)
                cuts,n = _add_cut(cuts,n,(sg*radii[j]*wp.length(c)-ac)/bc)
    cuts = _sort(cuts,n)
    total,errors = v12(wf(0.)),v12(wf(0.))
    visited = int(0)
    for part in range(n-1):
        stack_lo,stack_hi = v64(wf(0.)),v64(wf(0.))
        stack_lo[0] = cuts[part]
        stack_hi[0] = cuts[part+1]
        size = int(1)
        while size > 0:
            size -= 1
            lo,hi = stack_lo[size],stack_hi[size]
            if hi <= lo:
                continue
            fine = _panel(lo,hi,a,b,c,scales,radii,ubR,ubK,ubB,origin,origin_hi,gx,gw,1)
            coarse = _panel(lo,hi,a,b,c,scales,radii,ubR,ubK,ubB,origin,origin_hi,gx,gw,0)
            err,norm = wf(0.),wf(0.)
            for k in range(12):
                err = wp.max(err,wp.abs(fine[k]-coarse[k]))
                norm = wp.max(norm,wp.abs(fine[k]))
            visited += 1
            accept = err <= tolerance*(hi-lo+norm)
            if not accept and size < 62 and visited < max_panels and hi-lo > wf(1e-14):
                mid = wf(.5)*(lo+hi)
                stack_lo[size] = mid
                stack_hi[size] = hi
                stack_lo[size+1] = lo
                stack_hi[size+1] = mid
                size += 2
            else:
                if not accept:
                    status[row] = -1
                inner_coarse = _panel(lo,hi,a,b,c,scales,radii,ubR,ubK,ubB,origin,origin_hi,gx,gw,-1)
                for k in range(12):
                    total[k] += fine[k]
                    errors[k] += wp.abs(fine[k]-coarse[k])+wp.abs(fine[k]-inner_coarse[k])
    jac = area2/(h*h)
    for k in range(12):
        out[row,k] = background*plane[k]+sign*total[k]*scales[k]*jac
        estimate[row,k] = errors[k]*scales[k]*jac
        if status[row] < 0:
            out[row,k] = wf(wp.nan)


@lru_cache(None)
def _plan(kernel, device):
    ub = [(float(R),k,float(co)) for R,d in kernel_ubasis(kernel).items() for k,co in d.items()]
    radii = sorted(set(v[0] for v in ub))
    if len(radii)>2:
        raise NotImplementedError('Conditioned triangle device rule supports at most two kernel radii')
    origin_piece = get_kernel(kernel).pieces[0]
    roots,weights=[],[]
    for count in [8,16,32]:
        x,w=np.polynomial.legendre.leggauss(count)
        roots.extend((x+1)/2);weights.extend(w/2)
    plane = plane_channels(kernel)
    f = lambda a: wp.array(np.array(a,dtype=np.float64),dtype=wf,device=device)
    return (f(radii),f([v[0] for v in ub]),wp.array(np.array([v[1] for v in ub],dtype=np.int32),dtype=int,device=device),
            f([v[2] for v in ub]),f([float(co) for co in origin_piece[2]]),min(.125,float(origin_piece[1])),f(roots),f(weights),f(plane))


def plane_second_moment(kernel):
    """int_plane y_x² W = int_plane y_y² W, h=1; exact rational normalization."""
    return float(sum(Fraction(co)*(hi**(k+4)-lo**(k+4))/Fraction(k+4)
                     for lo,hi,coeffs in get_kernel(kernel).pieces for k,co in enumerate(coeffs)))


def plane_channels(kernel):
    """Exact rational whole-support mass shared by lam and both g1 diagonals."""
    mass = float(2*sum(Fraction(co)*(hi**(k+2)-lo**(k+2))/Fraction(k+2)
                       for lo,hi,coeffs in get_kernel(kernel).pieces for k,co in enumerate(coeffs)))
    second = plane_second_moment(kernel)
    return np.array([mass,0.,0.,0.,0.,mass,0.,0.,mass,second,0.,second])


def prepare(vertices, edges):
    """Static triangle metadata or None; no per-evaluation geometry host reads."""
    if len(vertices)!=3 or len(edges)!=3:
        return None
    v=vertices.detach().cpu().numpy()
    e={tuple(pair) for pair in edges.detach().cpu().tolist()}
    if e not in [{(0,1),(1,2),(2,0)},{(0,2),(2,1),(1,0)}]:
        return None
    orient = [0,1,2] if (0,1) in e else [0,2,1]
    vf=[[Fraction(float(co)) for co in p] for p in v]
    a,b,c=[vf[i] for i in orient]
    signed=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if not signed:
        return None
    # Short edge is the transverse direction. Preserve the integration sign.
    short=int(np.argmin([np.dot(v[(i+1)%3]-v[i],v[(i+1)%3]-v[i]) for i in range(3)]))
    return ([short,(short+2)%3,(short+1)%3],float(abs(signed)),float(1 if signed>0 else -1))


@lru_cache(None)
def _order(order, device):
    return wp.array(np.array(order,dtype=np.int32),dtype=int,device=device)


def channels(vertices, positions, supports, metadata, kernel, background=0, valid=None, tolerance=2e-12, max_panels=4096):
    """Device tensors (values[N,12], estimate[N,12], status[N]); normalized units."""
    if tolerance <= 0 or max_panels < 1:
        raise ValueError("positive tolerance and max_panels required")
    device=str(positions.device)
    values=torch.zeros((len(positions),12),dtype=torch.float64,device=device)
    estimate=torch.zeros_like(values)
    status=torch.zeros(len(positions),dtype=torch.int32,device=device)
    if metadata is not None and len(positions):
        order,area2,sign=metadata
        idx=_order(tuple(order),device)
        valid=torch.ones(len(positions),dtype=torch.int32,device=device) if valid is None else valid.to(torch.int32).contiguous()
        plan=_plan(kernel,device)
        wp.launch(_triangle_kernel,len(positions),device=device,inputs=[
            wp.from_torch(positions.to(torch.float64).contiguous(),dtype=v2),wp.from_torch(supports.to(torch.float64).contiguous(),dtype=wf),
            wp.from_torch(valid,dtype=wp.int32),wp.from_torch(vertices.to(torch.float64).contiguous(),dtype=v2),idx,
            wf(area2),wf(sign),wf(background),*plan,wf(tolerance),int(max_panels),
            wp.from_torch(values,dtype=wf),wp.from_torch(estimate,dtype=wf),wp.from_torch(status,dtype=wp.int32)])
        sync(device)
    return values,estimate,status
