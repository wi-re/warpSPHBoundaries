"""Refresh N1/N2/N3 evidence against precision-refined analytic references.

Does not overwrite earlier audits. Independent polar spot checks validate the
formula separately; precision refinement alone does not establish independence.
"""
import argparse
import json
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import mpmath as mp
import numpy as np

from warpSPHBoundaries.edge import core, geometry, oracle, np_fem
from warpSPHBoundaries.edge.triangle import integrate, difficult_mask
from warpSPHBoundaries.edge.kernels import kernel as get_kernel
from warpSPHBoundaries.validation.atlas import CHANNELS, ALPHAS, polygon_channels
from warpSPHBoundaries.validation.adversarial import (Measurement, polygon_cases, fem_channels,
    precision_polygon_reference as reference, reference_precision_estimate)
from warpSPHBoundaries.validation.robustness import save_measurements, collect_findings
from warpSPHBoundaries.validation.audit_report import generate, method_data
from warpSPHBoundaries.validation.report_view import accessible_report


def polar_check(vertices, point, kernel, target, dps=45):
    with mp.workdps(dps+20):
        pieces = [(lo, hi, {n-2:-n*c for n,c in enumerate(co) if n and c}) for lo,hi,co in get_kernel(kernel).pieces]
        values = [oracle.polar_moment(vertices, point, kernel, al, dps=dps) for al in ALPHAS]
        values += [oracle.polar_moment_profile(vertices, point, al, pieces, dps=dps)/mp.pi
                   for al in [(1,0),(0,1),(2,0),(1,1),(1,1),(0,2)]]
        return [float(abs(v-mp.mpf(str(t)))) for v,t in zip(values,target)]


def run(output, dpi=300):
    output = Path(output)
    if output.resolve() in [evidence_path('robustness').resolve(), evidence_path('accuracy-progress').resolve()]:
        raise ValueError('Use a separate conditioning checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    records, checks = [], []
    for kernel in ['cubic','w2','w4','w6']:
        for case in polygon_cases():
            folder = output/kernel/case.name
            folder.mkdir(parents=True, exist_ok=True)
            print(kernel,case.name,flush=True)
            points, vertices = case.probes(), case.vertices
            cache = folder/'reference.npz'
            if cache.exists():
                with np.load(cache) as a:
                    np.testing.assert_array_equal(a['points'], points)
                    np.testing.assert_array_equal(a['vertices'], vertices)
                    target, uncertainty = a['target'], a['uncertainty']
            else:
                coarse = reference(vertices, points, kernel, 45)
                target = reference(vertices, points, kernel, 70)
                uncertainty = reference_precision_estimate(target, coarse)
                np.savez_compressed(cache, target=target, uncertainty=uncertainty, points=points, vertices=vertices)
            uncertainty = reference_precision_estimate(target, target+uncertainty)
            closed = polygon_channels(vertices, points, kernel)
            measurements = [Measurement('closed-f64',closed,target,note='Current analytic polygon evaluator; exact geometry')]
            conditioned = closed.copy()
            estimate = np.full_like(conditioned, np.nan)
            selected = np.zeros(len(points),bool)
            if len(vertices)==3:
                v = np.broadcast_to(vertices,(len(points),3,2))
                selected = difficult_mask(v,points)
                for i in np.flatnonzero(selected):
                    r = integrate(vertices,points[i],kernel)
                    if not r.converged:
                        raise ArithmeticError(f'Outer quadrature failed: {kernel}/{case.name}/{i}')
                    conditioned[i],estimate[i] = r.moments,r.estimate
                # Execute the actual public FEM consumer, including its dispatch.
                hybrid = fem_channels(vertices,points,kernel,hybrid=True)
                measurements += [Measurement('P2-hybrid',hybrid,target,
                                 note='Current public weights_hybrid: conditioned area fallback on small/sliver triangles')]
            measurements += [Measurement('conditioned-cpu',conditioned,target,estimate=estimate,
                             note='Candidate: closed polygon identity plus conditioned area on small/sliver triangles. CPU only; not wired into Scene/CUDA.')]
            methods = save_measurements(folder,target,uncertainty,measurements,points=points,vertices=vertices,
                                        adaptive_selected=selected)
            records.append(dict(case=case.name,family=case.family,parameter=case.parameter,kernel=kernel,methods=methods))
            if case.name in ['triangle-size-1e-09','sliver-1e-07'] and kernel in ['cubic','w4']:
                i=44
                error=polar_check(vertices,points[i],kernel,target[i])
                checks.append(dict(case=case.name,kernel=kernel,point=points[i].tolist(),
                                   channels=list(CHANNELS),absolute_error=error,
                                   max_signal_relative_error=float(np.max(np.array(error)/np.max(abs(target),axis=0)))))
    summary=dict(channels=list(CHANNELS),dpi=dpi,records=records,specialized=[],representation=[],oracle=checks,
                 reference_descriptions={'polygon': 'Original float-input polygon geometry: all twelve moments from analytic identities at 70/45 decimal digits, stored in longdouble with a storage-rounding uncertainty floor. Independent polar checks cover four difficult queries across all moments. These precision-refined references supersede the earlier low-precision targets for this checkpoint only.'})
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (output/'independent-oracle-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (output/'coverage.json').write_text(json.dumps(dict(explicit_limits=[
        'All 21 original polygon fixtures, all original queries and four base kernels; no new GPU/SDF/lookup evaluation.',
        'Targets use exact float-input geometry and analytic identities at 70 versus 45 decimal digits, then longdouble storage.',
        'Independent polar integration checks four difficult-case/query/kernel combinations on all twelve channels.',
        'Reference precision refinement shares the analytic identities; polar coverage is finite, not a proof for all inputs.',
        'Conditioned CPU candidate keeps analytic polygon evaluation except for selected small/sliver triangles.',
        'P2-hybrid uses the actual public CPU dispatch; production Scene and CUDA remain unchanged.',
        'Adaptive error estimates are empirical, not certified bounds; no universal few-eps claim.']),indent=2)+'\n')
    finish(output, summary, dpi)


def finish(output, summary, dpi=300):
    """Render cached values with current reference uncertainty/display metadata."""
    output = Path(output)
    if not any(m['name']=='conditioned-cpu' for r in summary['records'] for m in r['methods']):
        raise ValueError('Not a triangle-conditioning checkpoint')
    summary['reference_descriptions'] = {'polygon': 'Original float-input polygon geometry: all twelve moments from analytic identities at 70/45 decimal digits, stored in longdouble with a storage-rounding uncertainty floor. Independent polar checks cover four difficult queries across all moments. These precision-refined references supersede the earlier low-precision targets for this checkpoint only.'}
    for row in summary['records']:
        file = output/row['kernel']/row['case']/'data.npz'
        with np.load(file) as a:
            payload = {key:a[key] for key in a.files}
        uncertainty = np.maximum(payload['reference_estimate'], 4*np.finfo(np.longdouble).eps*abs(payload['target']))
        if not np.array_equal(uncertainty, payload['reference_estimate']):
            payload['reference_estimate'] = uncertainty
            np.savez_compressed(file, **payload)
        for method in row['methods']:
            method['max_reference_estimate'] = [float(v) for v in uncertainty.max(0)]
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    collect_findings(summary,output)
    path=generate(output,dpi)
    stats={name:method_data(summary['records'],name) for name in ['closed-f64','conditioned-cpu','P2-hybrid']}
    counts={name:{key:list(d['outcomes'].values()).count(key) for key in ['within','exceeds','not run','nonfinite']} for name,d in stats.items()}
    (output/'checkpoint.json').write_text(json.dumps(dict(counts=counts,worst={k:d['worst'] for k,d in stats.items()},polar_checks=summary['oracle']),indent=2)+'\n')
    text=path.read_text();title,body=text.split('\n',1)
    intro=['## N1/N2/N3 conditioning checkpoint','',
           'Fresh CPU measurements against 70/45-digit analytic references. The conditioned route is a candidate for production polygon fallback; it is already used by the CPU triangle FEM hybrid. Scene/CUDA have not been changed by this checkpoint.','',
           '| Path | Within ε32 | Exceeds ε32 | Inapplicable |','|---|---:|---:|---:|']
    for name,c in counts.items():
        intro.append(f"| {name} | {c['within']} | {c['exceeds']} | {c['not run']} |")
    intro += ['', '[Independent polar checks](independent-oracle-checks.json) and [checkpoint](checkpoint.json).',
              'References here supersede the cached polygon targets for this checkpoint only. The baseline audit is preserved.','']
    path.write_text(title+'\n\n'+'\n'.join(intro)+body)
    accessible_report(path)
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=evidence_path('triangle-conditioning'))
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--report-only', action='store_true', help='Rebuild presentation and uncertainty metadata from cached values')
    args=p.parse_args()
    if args.report_only:
        finish(args.output,json.loads((Path(args.output)/'summary.json').read_text()),args.dpi)
    else:
        run(args.output,args.dpi)
