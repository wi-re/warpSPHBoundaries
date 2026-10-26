"""Bounded 3D integral grids, independent of the browser rendering backend.

Arrays are C-ordered (z, y, x), exported as little-endian float32/base64. The
sphere/plane adapters integrate piecewise polynomials with an exact-degree
Gauss rule; voxel convolution is explicitly a spatial quadrature approximation.
"""
import base64
from functools import lru_cache
import time

import numpy as np
from scipy.signal import fftconvolve

from ..curvbound.kernels import kernel
from .compute import KERNELS, number
from .terms3d import FIELDS3D, integrand
from ..mesh3d.radial import axisymmetric, select
from ..mesh3d.slender import ball_series

MODELS3D = {
    'sphere': {'label': 'Sphere · polynomial integral', 'description': 'Exact-degree radial shell integration of the solid sphere; float64 evaluation.'},
    'sdf': {'label': 'SDF · local plane', 'description': 'Three-dimensional half-space integral from exact geometry distance; local model error remains.'},
    'voxel': {'label': 'Voxel · FFT quadrature', 'description': 'Linear convolution of sampled solid occupancy with the 3D kernel; spatial quadrature error remains.'},
    'curvature': {'label': 'SDF · curvature O(2)', 'description': 'Order-two quadratic-graph correction to every moment integrand; sphere R/h ≥ 2.'},
    'ball_series': {'label': 'Small ball · order 4', 'description': 'Ball mean-value expansion through a⁴Δ²/280; R/h ≤ 0.3 and a single smooth kernel piece.'},
}
DEFAULT3D = dict(dimension=3, geometry='sphere', radius=1., width=2., height=2., depth=2.,
                 h=.6, resolution=49, padding=1.15, kernel='w4', field='m00',
                 models=['sphere', 'sdf', 'voxel'], supersampling=2, curvature_order=2, series_order=4)


def validate3d(payload):
    c = {**DEFAULT3D, **payload}
    for key, choices in (('geometry', ('sphere', 'box', 'tetrahedron')), ('kernel', KERNELS),
                         ('field', FIELDS3D)):
        if not isinstance(c[key], str) or c[key] not in choices:
            raise ValueError(f'Unknown 3D {key}')
    for key, lo, hi, integer in (('radius', .2, 5, False), ('width', .2, 6, False),
                               ('height', .2, 6, False), ('depth', .2, 6, False),
                               ('h', .15, 2, False), ('padding', 1, 2, False),
                               ('resolution', 17, 129, True), ('supersampling', 1, 2, True)):
        c[key] = number(c[key], key, lo, hi, integer)
    if not isinstance(c['models'], list) or not 1 <= len(c['models']) <= 3:
        raise ValueError('Choose one to three 3D models')
    if any(not isinstance(m, str) or m not in MODELS3D for m in c['models']):
        raise ValueError('Unknown 3D model')
    if len(set(c['models'])) != len(c['models']):
        raise ValueError('Choose distinct 3D models')
    if c['curvature_order'] not in (0,1,2) or c['series_order'] not in (0,2,4):
        raise ValueError('Unknown expansion order')
    c['dimension'] = 3
    return c


@lru_cache(None)
def gauss():
    # The largest supported profile has degree 11; shell geometry adds at most
    # three powers. Sixteen nodes integrate each split polynomial exactly in
    # exact arithmetic. Factored profiles avoid expanded-monomial cancellation.
    return np.polynomial.legendre.leggauss(16)


def profile(q, name):
    k = kernel(name)
    if name == 'cubic':
        shape = (1-q)**3 - 4*np.maximum(.5-q, 0)**3
    else:
        shape = (1-q)**k.n * sum(float(v)*q**p for p, v in k.poly)
    return np.where((q >= 0) & (q <= 1), shape*float(k.c3_pi)/np.pi, 0.)


def radial_integral(lo, hi, name, polynomial):
    lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
    result = np.zeros_like(lo)
    nodes, weights = gauss()
    cuts = (0., .5, 1.) if name == 'cubic' else (0., 1.)
    # Bound scratch memory even for an 81³ grid.
    flat = result.ravel()
    for begin in range(0, result.size, 4096):
        l, r = lo.ravel()[begin:begin+4096], hi.ravel()[begin:begin+4096]
        values = np.zeros_like(l)
        for a, b in zip(cuts[:-1], cuts[1:]):
            left, right = np.maximum(l, a), np.minimum(r, b)
            half = np.maximum(right-left, 0)/2
            q = (left+right)[:, None]/2 + half[:, None]*nodes
            # Pass the flat chunk to the geometric polynomial for array-valued D.
            values += half*np.sum(weights*profile(q, name)*polynomial(q, begin, len(l)), axis=1)
        flat[begin:begin+4096] = values
    return result


def planar_mass(distance, name):
    d = np.asarray(distance, float)
    a = np.minimum(abs(d), 1.)
    result = radial_integral(a, np.ones_like(a), name,
                             lambda q, start, count: 2*np.pi*q*(q-a.ravel()[start:start+count, None]))
    return np.where(d >= 0, result, 1-result)


def sphere_mass(points, radius, name):
    distance = np.linalg.norm(points, axis=1)
    # Unit support. Complete shells and partially intersected shells are disjoint.
    whole = radial_integral(np.zeros_like(distance), np.clip(radius-distance, 0, 1), name,
                            lambda q, *_: 4*np.pi*q*q)
    safe = np.where(distance > 0, distance, 1.)
    lower, upper = np.abs(distance-radius), np.minimum(1., distance+radius)
    partial = radial_integral(lower, upper, name,
        lambda q, start, count: np.pi*q*(radius**2-(safe[start:start+count, None]-q)**2)/safe[start:start+count, None])
    return whole + np.where(distance == 0, 0., partial)


def tetra_vertices(c):
    return np.array([[-1., -1., -1.], [1., 1., -1.], [1., -1., 1.], [-1., 1., 1.]]) * np.array([c['width'], c['height'], c['depth']])/2


def geometry3d(c):
    if c['geometry'] == 'sphere':
        return {'kind': 'sphere', 'radius': c['radius'], 'vertices': [], 'faces': []}
    if c['geometry'] == 'tetrahedron':
        return {'kind': 'tetrahedron', 'vertices': tetra_vertices(c).tolist(), 'faces': [[0,1,2], [0,3,1], [0,2,3], [1,3,2]]}
    vertices = [[a*c['width']/2, b*c['height']/2, d*c['depth']/2] for d in (-1,1) for b in (-1,1) for a in (-1,1)]
    return {'kind': 'box', 'vertices': vertices, 'faces': [[0,1,3], [0,3,2], [4,6,7], [4,7,5], [0,4,5], [0,5,1], [2,3,7], [2,7,6], [0,2,6], [0,6,4], [1,5,7], [1,7,3]]}


def signed_distance(points, c, return_normal=False):
    if c['geometry'] == 'sphere':
        length = np.linalg.norm(points, axis=-1)
        normals = points/np.where(length > 0, length, 1)[:,None]
        normals[length == 0] = (1,0,0)
        distance = length-c['radius']
        return (distance,normals) if return_normal else distance
    if c['geometry'] == 'box':
        q = abs(points)-np.array([c['width'], c['height'], c['depth']])/2
        outside = np.maximum(q, 0)*np.where(points >= 0, 1, -1)
        length = np.linalg.norm(outside,axis=-1)
        normals = outside/np.where(length > 0,length,1)[:,None]
        inside = length == 0
        ids = np.argmax(q[inside],axis=-1)
        normals[inside] = np.eye(3)[ids]*np.where(points[inside] >= 0,1,-1)
        distance = length+np.minimum(q.max(axis=-1),0)
        return (distance,normals) if return_normal else distance
    vertices = tetra_vertices(c)
    inside = np.ones(len(points), dtype=bool)
    best = np.full(len(points), np.inf)
    closest = np.zeros_like(points)
    nearest_plane = np.full(len(points), np.inf)
    inside_normal = np.zeros_like(points)
    for face in ((0,1,2), (0,3,1), (0,2,3), (1,3,2)):
        a,b,d = vertices[list(face)]
        u,v = b-a,d-a
        normal = np.cross(u,v)
        normal /= np.linalg.norm(normal)
        if np.dot(normal,a) < 0:
            normal = -normal
        relative = points-a
        plane = relative@normal
        inside &= plane <= 0
        nearer = abs(plane) < nearest_plane
        inside_normal[nearer] = normal
        nearest_plane[nearer] = abs(plane[nearer])
        # Orthogonal face projection, then the three edge projections.
        uu, uv, vv = u@u, u@v, v@v
        ru, rv = relative@u, relative@v
        alpha, beta = (vv*ru-uv*rv)/(uu*vv-uv*uv), (uu*rv-uv*ru)/(uu*vv-uv*uv)
        on_face = (alpha >= 0) & (beta >= 0) & (alpha+beta <= 1)
        dist2 = np.where(on_face, plane*plane, np.inf)
        target = points-plane[:,None]*normal
        for start,end in ((a,b),(b,d),(d,a)):
            edge = end-start
            t = np.clip((points-start)@edge/(edge@edge),0,1)
            delta = points-start-t[:,None]*edge
            edge_dist = np.sum(delta*delta,axis=1)
            update = edge_dist < dist2
            target[update] = (points-delta)[update]
            dist2 = np.minimum(dist2,edge_dist)
        update = dist2 < best
        closest[update] = target[update]
        best = np.minimum(best,dist2)
    distance = np.where(inside,-np.sqrt(best),np.sqrt(best))
    normals = (points-closest)/np.where(best > 0,np.sqrt(best),1)[:,None]
    normals[inside] = inside_normal[inside]
    return (distance,normals) if return_normal else distance


def voxel_mass(points, shape, spacing, c):
    # Treat query nodes as quadrature cell centres; fractional occupancy improves
    # cell-boundary coverage. All solids lie >=h away from the query box edge.
    occupancy = np.zeros(len(points))
    offsets = (0.,) if c['supersampling'] == 1 else (-.25, .25)
    for z in offsets:
        for y in offsets:
            for x in offsets:
                occupancy += signed_distance(points+spacing*np.array([x,y,z]),c) <= 0
    occupancy /= len(offsets)**3
    extent = int(np.ceil(c['h']/spacing))
    axis = np.arange(-extent,extent+1)*spacing/c['h']
    zz,yy,xx = np.meshgrid(axis,axis,axis,indexing='ij')
    offsets = np.stack([-xx,-yy,-zz],axis=-1)  # source-query, opposite convolution lag
    if c['field'] == 'grad_norm':
        components = [fftconvolve(occupancy.reshape(shape),integrand(offsets,c['kernel'],f'G{a}')*(spacing/c['h'])**3,mode='same') for a in 'xyz']
        return np.sqrt(sum(g*g for g in components)).ravel(), float((profile(np.linalg.norm(offsets,axis=-1),c['kernel'])*(spacing/c['h'])**3).sum())
    weights = integrand(offsets,c['kernel'],c['field'])*(spacing/c['h'])**3
    mass = fftconvolve(occupancy.reshape(shape),weights,mode='same')
    normalization = float((profile(np.linalg.norm(offsets,axis=-1),c['kernel'])*(spacing/c['h'])**3).sum())
    return mass.ravel(), normalization


def encode_volume(values):
    return base64.b64encode(np.asarray(values,dtype='<f4').tobytes()).decode('ascii')


def compute3d(payload):
    started = time.perf_counter()
    c = validate3d(payload)
    n = c['resolution']
    half = c['radius'] if c['geometry'] == 'sphere' else max(c['width'],c['height'],c['depth'])/2
    half += c['padding']*c['h']
    axis = np.linspace(-half,half,n)
    spacing = float(axis[1]-axis[0])
    zz,yy,xx = np.meshgrid(axis,axis,axis,indexing='ij')
    points = np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])
    distance, normals = signed_distance(points,c,True)
    results = []
    for model in c['models']:
        tick = time.perf_counter()
        notes = []
        if model in ('sphere','curvature','ball_series') and c['geometry'] != 'sphere':
            results.append(dict(model=model,label=MODELS3D[model]['label'],error='Sphere polynomial model requires sphere geometry'))
            continue
        if model == 'curvature' and c['radius']/c['h'] < 2:
            results.append(dict(model=model,label=MODELS3D[model]['label'],error='Curvature model requires R/h ≥ 2; this is a domain restriction, not an error certificate'))
            continue
        if model == 'ball_series' and c['radius']/c['h'] > .3:
            results.append(dict(model=model,label=MODELS3D[model]['label'],error='Small-ball model requires R/h ≤ 0.3'))
            continue
        if model == 'sphere':
            values = select(axisymmetric(distance/c['h'],c['kernel'],radius=c['radius']/c['h']),normals,c['field'])
            notes.append('Float64 shell integrals of the selected moment or analytic kernel derivative; no mass-grid finite differences. 16-node exact-degree radial Gauss rule split at knots and shell intersections.')
        elif model in ('sdf','curvature'):
            values = select(axisymmetric(distance/c['h'],c['kernel'],curvature=c['h']/c['radius'] if model=='curvature' else None,curvature_order=c['curvature_order']),normals,c['field'])
            notes.append('Exact geometry distance and nearest normal; deterministic normal at medial ties. Local geometry model error remains. Selected derivatives act on the kernel integrand.')
            if model == 'curvature':
                notes.append(f'Quadratic graph expansion through curvature order {c["curvature_order"]}, applied consistently to each integrand. Umbilic sphere only; omitted graph/higher-order terms remain.')
        elif model == 'ball_series':
            values = np.array([ball_series(p/c['h'],c['radius']/c['h'],c['kernel'],c['field'],c['series_order']) for p in points])
            notes.append('Ball mean-value expansion through a⁴Δ²/280. Samples crossing a kernel knot/support or including the radial origin remain missing; no fallback substitutes zero.')
        else:
            values, normalization = voxel_mass(points,(n,n,n),spacing,c)
            notes += [f'Voxel quadrature: {c["supersampling"]}³ occupancy samples/cell; linear zero-padded FFT convolution.',
                      f'Sampled kernel mass = {normalization:.8g}; no renormalization or clipping. Refine the grid to assess spatial quadrature error.']
            if spacing/c['h'] > .25:
                notes.append('Coarse kernel sampling: grid spacing exceeds 0.25 h; increase resolution or support.')
        finite = np.isfinite(values)
        if not finite.any():
            results.append(dict(model=model,label=MODELS3D[model]['label'],error='No samples lie in this model’s validity domain'))
            continue
        results.append(dict(model=model,label=MODELS3D[model]['label'],values=encode_volume(values),
                            min=float(np.min(values[finite])),max=float(np.max(values[finite])),missing=int((~finite).sum()),notes=notes,
                            milliseconds=round((time.perf_counter()-tick)*1000,1)))
    return dict(schema_version=1,dimension=3,encoding='base64-f32-le',order='zyx',shape=[n,n,n],
                config=c,x=axis.tolist(),y=axis.tolist(),z=axis.tolist(),geometry=geometry3d(c),
                distance=encode_volume(distance),results=results,
                milliseconds=round((time.perf_counter()-started)*1000,1))
