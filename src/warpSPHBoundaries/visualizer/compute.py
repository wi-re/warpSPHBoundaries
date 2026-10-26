"""Numerical adapters independent of HTTP and the static canvas client.

Coordinates are physical; returned moment channels use support units throughout.
No display clipping is applied to numerical values or signed differences.
"""
from functools import lru_cache
from types import SimpleNamespace
import time

import numpy as np
from scipy.ndimage import map_coordinates

from ..edge import np2d
from ..validation.atlas import CHANNELS, disk_reference
from ..validation.distance_only import exact_distance_normal, halfplane_channels

MODELS = {
    "mesh": {"label": "Triangle mesh", "description": "Sum of exact polygon-edge integrals over triangles."},
    "exact": {"label": "Exact polygon", "description": "Closed-form edge reduction over the represented polygon."},
    "rectangle": {"label": "Built-in rectangle", "description": "Four-edge closed-form rectangle, using the same NumPy edge engine as the mesh."},
    "sdf": {"label": "SDF · distance only", "description": "Local half-plane from signed distance; corner and curvature model errors remain."},
    "curvature": {"label": "SDF · curvature table", "description": "Existing Tier3 order-two expansion, circle only, R/h ≥ 2."},
    "disk": {"label": "Circle reference", "description": "Ray/disk Gauss quadrature with analytic radial integration; not an exact closed form."},
}
KERNELS = {"w2": "Wendland C2", "w4": "Wendland C4", "w6": "Wendland C6", "cubic": "Cubic spline"}
FIELDS = {key: key for key in CHANNELS}
FIELDS["grad_norm"] = "|h ∇λ|"
DEFAULT = dict(geometry="rectangle", width=2., height=2., radius=2., angle=0.,
               vertices=[[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]],
               segments=16, kernel="w4", h=.65, resolution=81, padding=1.15,
               field="m00", models=["mesh", "rectangle", "sdf"],
               sdf_source="analytic", sdf_spacing=.12, quadrature=24, curvature_order=2, series_order=4)


def number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high or (integer and int(value) != value):
        raise ValueError(f"{name} must be {'an integer ' if integer else ''}in [{low}, {high}]")
    return int(value) if integer else float(value)


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object")
    c = {**DEFAULT, **payload}
    for key, choices in (("geometry", ("rectangle", "triangle", "circle")), ("kernel", KERNELS),
                         ("field", FIELDS), ("sdf_source", ("analytic", "sampled"))):
        if not isinstance(c[key], str) or c[key] not in choices:
            raise ValueError(f"Unknown {key}")
    for key, low, high, integer in (("width", .1, 8, False), ("height", .1, 8, False),
                                  ("radius", .15, 8, False), ("h", .1, 3, False),
                                  ("angle", -180, 180, False), ("resolution", 17, 513, True),
                                  ("padding", 1, 2, False), ("segments", 8, 64, True),
                                  ("sdf_spacing", .03, .5, False), ("quadrature", 8, 64, True)):
        c[key] = number(c[key], key, low, high, integer)
    if c["curvature_order"] not in (0, 1, 2) or c["series_order"] not in (0, 2, 4):
        raise ValueError("Unknown expansion order")
    if not isinstance(c["models"], list) or not 1 <= len(c["models"]) <= 4:
        raise ValueError("Choose one to four models")
    if any(not isinstance(m, str) or m not in MODELS for m in c["models"]):
        raise ValueError("Unknown model")
    if len(set(c["models"])) != len(c["models"]):
        raise ValueError("Choose distinct models")
    try:
        v = np.asarray(c["vertices"], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Invalid vertices") from None
    if v.shape != (4, 2) or not np.isfinite(v).all() or np.abs(v).max() > 8:
        raise ValueError("Provide four finite vertices with coordinates in [-8, 8]")
    c["vertices"] = v.tolist()
    return c


def geometry(c):
    if c["geometry"] == "circle":
        a = np.arange(c["segments"]) * 2*np.pi/c["segments"]
        v = c["radius"] * np.column_stack([np.cos(a), np.sin(a)])
        return v, [np.array([[0., 0.], v[i], v[(i+1) % len(v)]]) for i in range(len(v))]
    if c["geometry"] == "rectangle":
        v = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]) * [c["width"]/2, c["height"]/2]
    else:
        v = np.array(c["vertices"])[:3]
    a = np.deg2rad(c["angle"])
    v = v @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    a, b = v[1]-v[0], v[2]-v[0]
    area = a[0]*b[1]-a[1]*b[0]
    if abs(area) < 1e-7:
        raise ValueError("Triangle vertices must not be collinear")
    triangles = [v] if len(v) == 3 else [v[[0, 1, 2]], v[[0, 2, 3]]]
    return v, triangles


def signed(c, vertices, points):
    if c["geometry"] == "circle":
        r = np.linalg.norm(points, axis=1)
        return r-c["radius"], points/np.maximum(r[:, None], 1e-30)
    case = SimpleNamespace(kind="polygon", vertices=vertices)
    return tuple(np.asarray(x, dtype=float) for x in exact_distance_normal(case, points))


def sdf(c, vertices, points):
    d, n = signed(c, vertices, points)
    k = np.full(len(points), 1/c["radius"] if c["geometry"] == "circle" else 0.)
    if c["sdf_source"] == "analytic":
        return d/c["h"], n, k*c["h"]
    spacing = c["sdf_spacing"]*c["h"]
    origin = points.min(0)-3*spacing
    shape = np.ceil((points.max(0)-origin)/spacing).astype(int)+4
    if np.prod(shape) > 2_000_000:
        raise ValueError("SDF lattice too large; increase SDF spacing")
    xx, yy = np.meshgrid(*(origin[j]+np.arange(shape[j])*spacing for j in (0, 1)), indexing="ij")
    field = signed(c, vertices, np.column_stack([xx.ravel(), yy.ravel()]))[0].reshape(tuple(shape))
    gx, gy = np.gradient(field, spacing, edge_order=2)
    gxx, gxy = np.gradient(gx, spacing, edge_order=2)
    _, gyy = np.gradient(gy, spacing, edge_order=2)
    uv = ((points-origin)/spacing).T
    sample = lambda f: map_coordinates(f, uv, order=1, mode="nearest")
    d, nx, ny = sample(field), sample(gx), sample(gy)
    norm = np.hypot(nx, ny)
    n = np.column_stack([nx, ny])/np.maximum(norm[:, None], 1e-30)
    n[norm < 1e-6] = np.nan
    level_k = (ny*ny*sample(gxx)-2*nx*ny*sample(gxy)+nx*nx*sample(gyy))/np.maximum(norm**3, 1e-30)
    denominator = 1-d*level_k
    k = np.divide(level_k, denominator, out=np.full_like(d, np.nan), where=np.abs(denominator) > .05)
    return d/c["h"], n, k*c["h"]


def component(channels, field):
    return np.linalg.norm(channels[:, 6:8], axis=1) if field == "grad_norm" else channels[:, CHANNELS.index(field)]


def polygon_value(vertices, points, c, *, gradient_vector=False):
    values = []
    # Preserve shared triangle-edge consistency near collinear query/vertex rays.
    # Converting *before* division matters: float64 normalization can put the
    # same seam on opposite sides of winding/atan tests in adjacent triangles.
    dtype = np.longdouble
    v = np.asarray(vertices, dtype=dtype)/dtype(c["h"])
    for start in range(0, len(points), 512):
        p = np.asarray(points[start:start+512], dtype=dtype)/dtype(c["h"])
        batch = np.broadcast_to(v, (len(p), *v.shape))
        f = c["field"]
        if f == "m00":
            result = np2d.value(batch, p, c["kernel"], dtype=dtype)
        elif f.startswith("m"):
            result = np2d.moment(batch, p, c["kernel"], tuple(map(int, f[1:])), dtype=dtype)
        else:
            alpha = (0, 0) if f in ("Gx", "Gy", "grad_norm") else ((1, 0) if f[2] == "x" else (0, 1))
            g = np2d.grad_moment(batch, p, c["kernel"], alpha, dtype=dtype)
            result = (g if gradient_vector else np.linalg.norm(g, axis=1)) if f == "grad_norm" else g[:, 0 if f[-1] == "x" else 1]
        values.append(result)
    return np.concatenate(values)


@lru_cache(maxsize=4)
def tier3(kernel):
    from ..scene.implicitBodies import Tier3
    return Tier3(kernel, "cpu")


def evaluate_model(model, c, vertices, triangles, points, jets):
    if model in ("mesh", "exact", "rectangle"):
        if model == "rectangle" and c["geometry"] != "rectangle":
            raise ValueError("Built-in rectangle requires rectangle geometry")
        if model == "mesh":
            vector = c["field"] == "grad_norm"
            value = sum(polygon_value(t, points, c, gradient_vector=vector) for t in triangles)
            if vector:
                value = np.linalg.norm(value, axis=1)
        else:
            value = polygon_value(vertices, points, c)
        notes = ["CPU edge reduction in NumPy longdouble; results serialized as float64."]
        if np.finfo(np.longdouble).eps == np.finfo(np.float64).eps:
            notes.append("On this platform longdouble has float64 precision; seam conditioning may be weaker.")
        if c["geometry"] == "circle":
            notes.append(f"Inscribed {len(vertices)}-gon approximates the circle.")
        return value, notes
    if model in ("sdf", "curvature"):
        d, n, k = jets
        notes = ["Analytic distance/normal inputs." if c["sdf_source"] == "analytic" else f"Bilinear sampled SDF; lattice spacing {c['sdf_spacing']:g} h; finite-difference jets."]
        if model == "sdf":
            if c["field"] == "m00":
                # Scalar mass depends on d alone, even at undefined normals.
                plane = np.array([[-4., -4.], [4., -4.], [4., 0.], [-4., 0.]])
                local = {**c, "h": 1.}
                value = polygon_value(plane, np.column_stack([np.zeros_like(d), np.clip(d, -2, 2)]), local)
                value = np.where(d >= 1, 0., np.where(d <= -1, 1., value))
            else:
                value = component(halfplane_channels(d, n, c["kernel"]), c["field"])
            notes.append("Local-plane approximation; misses corners and curvature.")
            return value, notes
        if c["geometry"] != "circle":
            raise ValueError("Smooth curvature model is unavailable at polygon corners; choose Circle")
        if c["radius"]/c["h"] < 2:
            raise ValueError("Curvature model requires R/h ≥ 2 in this visualizer")
        import torch
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float64)
        values = tier3(c["kernel"]).curvature_moments(tensor(d), tensor(n), tensor(k)).numpy()
        if c['curvature_order'] == 0:
            values = halfplane_channels(d, n, c['kernel'])
        elif c['curvature_order'] == 1:
            reflected = tier3(c['kernel']).curvature_moments(tensor(d), tensor(n), tensor(-k)).numpy()
            base = tier3(c['kernel']).curvature_moments(tensor(d), tensor(n), tensor(np.zeros_like(k))).numpy()
            values = base+(values-reflected)/2
        notes.append(f"Local curvature order {c['curvature_order']}; no certified error tolerance. Circle curvature variation is zero.")
        return component(values, c["field"]), notes
    if c["geometry"] != "circle":
        raise ValueError("Circle reference requires circle geometry")
    values = disk_reference(points/c["h"], c["radius"]/c["h"], c["kernel"], c["quadrature"])
    return np.asarray(component(values, c["field"]), float), [f"{c['quadrature']}-node angular quadrature; analytic radial integration. Refinement is empirical, not a bound."]


def finite_list(a):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(a).ravel()]


def compute(payload):
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object")
    dimension = payload.get('dimension', 2)
    if isinstance(dimension, bool) or dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    if dimension == 3:
        from .compute3d import compute3d
        return compute3d(payload)
    start = time.perf_counter()
    c = validate(payload)
    vertices, triangles = geometry(c)
    # Exact circle bounds rather than its inscribed mesh bounds.
    lo = np.full(2, -c["radius"]) if c["geometry"] == "circle" else vertices.min(0)
    hi = -lo if c["geometry"] == "circle" else vertices.max(0)
    centre = (lo+hi)/2
    half = max(hi-lo)/2 + c["padding"]*c["h"]
    x = np.linspace(centre[0]-half, centre[0]+half, c["resolution"])
    y = np.linspace(centre[1]-half, centre[1]+half, c["resolution"])
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    distance = signed(c, vertices, points)[0]
    jets = sdf(c, vertices, points) if any(m in c["models"] for m in ("sdf", "curvature")) else None
    results = []
    for model in c["models"]:
        t = time.perf_counter()
        try:
            values, notes = evaluate_model(model, c, vertices, triangles, points, jets)
            finite = values[np.isfinite(values)]
            results.append(dict(model=model, label=MODELS[model]["label"], values=finite_list(values),
                                min=float(finite.min()) if len(finite) else None,
                                max=float(finite.max()) if len(finite) else None,
                                missing=int((~np.isfinite(values)).sum()), notes=notes,
                                milliseconds=round((time.perf_counter()-t)*1000, 1)))
        except ValueError as exc:
            results.append(dict(model=model, label=MODELS[model]["label"], error=str(exc)))
    return dict(schema_version=1, dimension=2, config=c, x=x.tolist(), y=y.tolist(),
                vertices=vertices.tolist(), triangles=[t.tolist() for t in triangles],
                distance=finite_list(distance), results=results,
                milliseconds=round((time.perf_counter()-start)*1000, 1))
