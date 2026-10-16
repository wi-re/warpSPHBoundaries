"""Refresh C1/C2 evidence without overwriting the pre-fix approximation audit."""
import argparse
import hashlib
import json
from pathlib import Path

from warpSPHBoundaries.validation.paths import evidence_path

import numpy as np

from warpSPHBoundaries.scene.scene import SurfaceRep
from warpSPHBoundaries.validation.adversarial import Measurement, SUPPORTED, scene_channels
from warpSPHBoundaries.validation.audit_report import generate, method_data, collections
from warpSPHBoundaries.validation.robustness import save_measurements, collect_findings
from warpSPHBoundaries.validation.specialized_audit import arc_measurements, register_kernels


def _stats(data):
    return dict(cases=len(data['cases']), within_eps32=list(data['outcomes'].values()).count('within'),
                nonfinite_cases=list(data['outcomes'].values()).count('nonfinite'),
                worst_relative=data['worst'],
                max_absolute=float(np.nanmax(data['matrix']['absolute'])))


def run(baseline=evidence_path('robustness'), output=evidence_path('accuracy-progress'), device='cuda:0', dpi=300):
    baseline, output = Path(baseline), Path(output)
    if baseline.resolve() == output.resolve():
        raise ValueError('Use a separate checkpoint folder; preserve the baseline audit')
    output.mkdir(parents=True, exist_ok=True)
    source = baseline/'summary.json'
    old = json.loads(source.read_text())
    records = []
    for row in old['records']:
        relative = Path(row['kernel'])/row['case']
        print('C1:', row['case'], row['kernel'], flush=True)
        with np.load(baseline/relative/'data.npz') as a:
            target, uncertainty = a['target'], a['reference_estimate']
            points, vertices = a['points'], a['vertices']
        measurements = []
        for name, fused, stable in (('scene-polygon', False, False),
                                     ('fused-mono', True, False), ('fused-cheb', True, True)):
            values = scene_channels(SurfaceRep.polygon(vertices), points, row['kernel'], device,
                                    fused=fused, stable=stable)
            measurements.append(Measurement(name, values, target, supported=SUPPORTED,
                                            note='C1 checkpoint: current polygon execution against cached original-geometry reference'))
        folder = output/relative
        folder.mkdir(parents=True, exist_ok=True)
        methods = save_measurements(folder, target, uncertainty, measurements, points=points, vertices=vertices)
        records.append(dict(row, methods=methods))
    specialized = []
    register_kernels()
    for study in old['specialized']:
        if study['study'] != 'disk':
            continue
        rows = []
        for row in study['rows']:
            radius = row['parameter']
            print('C2:', study['kernel'], radius, flush=True)
            relative = Path('specialized')/('disk-'+study['kernel'])/f'R-{radius:.12g}'
            with np.load(baseline/relative/'data.npz') as a:
                target, uncertainty, points = a['target'], a['reference_estimate'], a['points']
            methods = arc_measurements(points, radius, study['kernel'], target)
            folder = output/relative
            folder.mkdir(parents=True, exist_ok=True)
            summaries = save_measurements(folder, target, uncertainty, methods, points=points)
            rows.append(dict(row, methods=summaries))
        specialized.append(dict(study, rows=rows))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    new = dict(old, records=records, specialized=specialized, device=device, dpi=dpi,
               checkpoint='C1 polygon predicates and C2 circle flux; other schemes not rerun',
               baseline_summary_sha256=digest)
    (output/'summary.json').write_text(json.dumps(new, indent=2, allow_nan=False)+'\n')
    for filename in ('representation.json', 'independent-oracle-checks.json'):
        if (baseline/filename).exists():
            (output/filename).write_bytes((baseline/filename).read_bytes())
    coverage = dict(updated_paths=['scene-polygon', 'fused-mono', 'fused-cheb', 'arc-24', 'arc-48'],
                    explicit_limits=[
                        'This checkpoint reruns only C1/C2 paths, not SDF selection, FEM hybrid, lookup tables, strip series or the complete audit.',
                        'Original cached reference integrals are retained; their uncertainty and conditioning limits still apply.',
                        'Circle boundary regressions separately use 384/768-node independent polar reference refinement.',
                        'Passing boundary-convention checks does not establish epsilon32 accuracy on every small moment.',
                        'No production performance benchmark or broad float32 accuracy claim is made.'])
    (output/'coverage.json').write_text(json.dumps(coverage, indent=2)+'\n')
    collect_findings(new, output)
    comparisons = []
    before, after = collections(old), collections(new)
    for kind, names in (('polygon', ('fused-mono', 'fused-cheb')),
                         ('disk', ('arc-24', 'arc-48'))):
        for name in names:
            comparisons.append(dict(study=kind, method=name,
                                    before=_stats(method_data(before[kind], name)),
                                    after=_stats(method_data(after[kind], name))))
    (output/'checkpoint.json').write_text(json.dumps(dict(
        device=device, baseline_summary_sha256=digest, comparisons=comparisons), indent=2, allow_nan=False)+'\n')
    report = generate(output, dpi)
    intro = ['## C1/C2 progress checkpoint', '',
             'Only the named production polygon and circle-arc paths were re-evaluated. '
             'The original survey is preserved. Reference precision limits remain in force.', '',
             '| Path | Before: cases within ε32 | After: cases within ε32 | Before max absolute error | After max absolute error |',
             '|---|---:|---:|---:|---:|']
    for c in comparisons:
        b, a = c['before'], c['after']
        intro.append(f"| {c['study']} / {c['method']} | {b['within_eps32']}/{b['cases']} | "
                     f"{a['within_eps32']}/{a['cases']} | {b['max_absolute']:.3g} | {a['max_absolute']:.3g} |")
    intro += ['', 'These counts use every implemented channel and tested kernel; '
              'they are broader than the boundary-convention regression gates. '
              'Remaining numerical-accuracy failures belong in the conditioning/reference workstreams.', '',
              '[Machine-readable checkpoint](checkpoint.json).', '']
    text = report.read_text()
    header, body = text.split('\n', 1)
    report.write_text(header+'\n\n'+'\n'.join(intro)+body)
    from warpSPHBoundaries.validation.report_view import accessible_report
    accessible_report(report)
    return comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default=evidence_path('robustness'))
    p.add_argument('--output', default=evidence_path('accuracy-progress'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dpi', type=int, default=300)
    a = p.parse_args()
    print(json.dumps(run(a.baseline, a.output, a.device, a.dpi), indent=2))


if __name__ == '__main__':
    main()
