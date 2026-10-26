"""Compare constant-block volume/area with Strobl et al.'s independent overlap.

Optional dependency: pip install overlap (authors' MIT-licensed implementation).
No dependency is added to the runtime package. Use --overlap-path for an
isolated --target install. JSON includes every geometry and the package version.
The two analytic algorithms are compared, without treating either float64
output or precision refinement as an exact independent error certificate.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import mpmath as mp
import numpy as np

from warpSPHBoundaries.mesh3d import Tetrahedron
from warpSPHBoundaries.mesh3d.oracle import constant_overlap
from warpSPHBoundaries.validation.paths import evidence_path


def cases(random_count=200, seed=20261022):
    unit = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
    for label, q, r in [
        ("vertex", (0, 0, 0), .2), ("edge", (.4, 0, 0), .1),
        ("face", (.3, .3, 0), .1), ("inside", (.2, .2, .2), .05),
        ("engulfed", (.2, .2, .2), 2.), ("empty", (-2, 0, 0), .1),
        ("tangent", (-.2, 0, 0), .2), ("near-tangent", (-.1999999, 0, 0), .2),
        ("generic-cut", (.2, .2, .2), .3),
    ]:
        yield label, unit, np.array(q, float), r
    rng = np.random.default_rng(seed)
    for i in range(random_count):
        scale = 10.**rng.uniform(-6, 6)
        aspect = 10.**rng.uniform(-8, 0) if i % 4 == 0 else 1.
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        transform = scale*np.diag([1, 1, aspect])@rotation
        vertices = unit@transform
        if np.linalg.det((vertices[1:]-vertices[0]).T) < 0:
            vertices[[1, 2]] = vertices[[2, 1]]
        query = rng.uniform(-.4, .9, 3)@transform
        radius = scale*10.**rng.uniform(-2, 1)
        yield f"random-{i:04d}", vertices, query, radius


def reference(vertices, query, radius, dps):
    with mp.workdps(dps):
        t = Tetrahedron(vertices, query)
        r = mp.mpf(float(radius))
        vol = t.block_moment((0, 0, 0), 0, r)
        area = sum(f.block_moment((0, 0, 0), 0, r) for f in t.faces)
        return vol, area


def run(output, random_count=200, seed=20261022, oracle_count=5):
    import overlap
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for label, vertices, query, radius in cases(random_count, seed):
        tet = overlap.Tetrahedron(vertices)
        sphere = overlap.Sphere(query, radius)
        external = (float(overlap.overlap_volume(sphere, tet)), float(overlap.overlap_area(sphere, tet)[-1]))
        coarse = reference(vertices, query, radius, 45)
        refined = reference(vertices, query, radius, 75)
        vol_scale = min(tet.volume, 4*np.pi*radius**3/3)
        area_scale = min(tet.surface_area, 4*np.pi*radius**2)
        metrics = []
        for a, b, c, scale in zip(refined, external, coarse, (vol_scale, area_scale)):
            with mp.workdps(80):
                delta = abs(a-mp.mpf(b))
                metrics.append(dict(reference=mp.nstr(a, 65), overlap=b,
                                    absolute_difference=float(delta),
                                    scale=float(scale), scaled_difference=float(delta/scale),
                                    relative_difference=float(delta/abs(a)) if a else None,
                                    precision_difference=float(abs(a-c))))
        records.append(dict(case=label, vertices=vertices.tolist(), query=query.tolist(), radius=radius,
                            volume=metrics[0], total_face_area=metrics[1]))
    # Resolve the largest discrepancies with a third, positive-integrand path.
    for row in sorted(records, key=lambda r: r["volume"]["scaled_difference"], reverse=True)[:oracle_count]:
        with mp.workdps(80):
            v, e = constant_overlap(row["vertices"], row["query"], row["radius"])
            v2, e2 = constant_overlap(row["vertices"], row["query"], row["radius"], tolerance=2e-13)
            row["independent_quadrature"] = dict(value=mp.nstr(v2, 65), estimate=float(e2),
                                                 refinement_difference=float(abs(v2-v)),
                                                 coarse_estimate=float(e),
                                                 analytic_difference=float(abs(v2-mp.mpf(row["volume"]["reference"]))),
                                                 overlap_difference=float(abs(v2-row["volume"]["overlap"])))
        print(f"independent clipped-simplex check: {row['case']}", flush=True)
    summary = dict(paper="https://doi.org/10.1016/j.jcp.2016.02.003",
                   implementation="https://github.com/severinstrobl/overlap",
                   overlap_version=version("overlap"), seed=seed, count=len(records),
                   reference_dps=[45, 75], records=records)
    (output/"comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    for field in ("volume", "total_face_area"):
        worst = max(records, key=lambda row: row[field]["scaled_difference"])
        print(f"{field}: worst scaled difference {worst[field]['scaled_difference']:.3e} ({worst['case']})", flush=True)
    print(f"{len(records)} cases; {output/'comparison.json'}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=evidence_path("mesh3d-overlap"))
    parser.add_argument("--overlap-path", type=Path)
    parser.add_argument("--random-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--oracle-count", type=int, default=5)
    args = parser.parse_args()
    if args.overlap_path:
        sys.path.insert(0, str(args.overlap_path.resolve()))
    run(args.output, args.random_count, args.seed, args.oracle_count)
