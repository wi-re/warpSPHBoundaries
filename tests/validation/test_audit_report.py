"""Report classification and error attribution must not hide failed coverage."""
import hashlib
import json
import re

import numpy as np
import pytest

from warpSPHBoundaries.validation.atlas import CHANNELS
from warpSPHBoundaries.validation.audit_report import (
    collections, method_data, signed_budget, budget_data, generate, scheme_info)
from warpSPHBoundaries.validation.error_colors import EPS32


def measurement(name='method', error=0., supported=CHANNELS, failures=0, domain=2):
    values = [error if ch in supported else None for ch in CHANNELS]
    return dict(name=name, supported_channels=list(supported),
                nonfinite_implemented_values=failures, in_domain_queries=domain,
                max_absolute_error=values.copy(), max_error_over_signal_scale=values.copy(),
                signal_scale=[1.]*12, status='evaluated')


def row(case, kernel='w4', methods=None):
    return dict(case=case, kernel=kernel, family='test', parameter=1.,
                methods=[measurement()] if methods is None else methods,
                source=f'{kernel}/{case}/data.npz')


def test_worst_kernel_aggregation_preserves_missing_cases_and_failure_status():
    rows = [row('new-shape', methods=[measurement(error=EPS32/2)]),
            row('new-shape', 'w2', [measurement(error=2*EPS32)]),
            row('collapsed', methods=[measurement(error=0., failures=2)]),
            row('unsupported', methods=[measurement(supported=('m00',))]),
            row('rejected', methods=[measurement(error=None, supported=('m00',), domain=0)]),
            row('not-applicable', methods=[])]
    data = method_data(rows, 'method')
    assert data['outcomes'] == dict([('new-shape', 'exceeds'), ('collapsed', 'nonfinite'),
                                    ('unsupported', 'within'), ('rejected', 'unassessed'),
                                    ('not-applicable', 'not run')])
    assert data['matrix']['relative'][0, 0] == 2*EPS32
    assert data['matrix']['relative'][0, 2] == 0
    assert np.isnan(data['matrix']['relative'][1:, 2:]).all()
    assert data['worst']['kernel'] == 'w2'


def test_close_tier_threshold_parameters_are_distinct_and_new_studies_are_discovered():
    summary = dict(records=[], specialized=[
        dict(study='disk', kernel='w4', rows=[dict(parameter=p, methods=[]) for p in (.19999999, .2, .20000001)]),
        dict(study='new-geometry', kernel='w4', rows=[dict(parameter=3., case='custom', source='custom/data.npz', methods=[])])])
    groups = collections(summary)
    assert len({r['case'] for r in groups['disk']}) == 3
    assert groups['disk'][0]['source'] == 'specialized/disk-w4/R-0.19999999/data.npz'
    assert groups['new-geometry'][0]['source'] == 'custom/data.npz'
    assert scheme_info('new-scheme')[0] == 'expectation unspecified'


@pytest.mark.parametrize('lost', [False, True])
def test_signed_sdf_geometry_split_closes_even_with_cancelling_errors(lost):
    target = np.arange(24, dtype=np.longdouble).reshape(2, 12)+1
    contour = np.zeros_like(target) if lost else target+3
    actual = target-2
    a = dict(target=target, **{'SDF-contour': contour, 'SDF-forced': actual})
    b = signed_budget(a, 'SDF-forced')
    np.testing.assert_array_equal(b['geometry'], contour-target)
    np.testing.assert_array_equal(b['local + evaluation'], actual-contour)
    np.testing.assert_array_equal(b['total'], b['geometry']+b['local + evaluation'])
    assert 'numerical' not in b
    assert set(signed_budget(a, 'SDF-contour')) == {'total', 'geometry'}
    assert set(signed_budget(dict(target=target, unknown=actual), 'unknown')) == {'total'}


def test_budget_reduction_respects_declared_domains_and_unsupported_channels(tmp_path):
    arrays = dict(target=np.ones((2, 12)), method=np.ones((2, 12)),
                  method_numerical_error=np.full((2, 12), 100.),
                  method_model_error=np.full((2, 12), 200.),
                  method_domain=np.array([False, True]))
    arrays['method_numerical_error'][1, 0] = .01
    arrays['method_model_error'][1, 0] = .02
    r = row('domain', methods=[measurement(supported=('m00',))])
    r['source'] = 'data.npz'
    np.savez(tmp_path/'data.npz', **arrays)
    data = method_data([r], 'method')
    b = budget_data([r], 'method', data, tmp_path, {})
    assert b['numerical'][0, 0] == .01
    assert b['model / geometry'][0, 0] == .02
    assert np.isnan(b['numerical'][1:]).all()


def test_generated_report_paginates_new_cases_and_preserves_cached_evidence(tmp_path):
    rows = []
    for i in range(3):
        r = row(f'future-shape-{i}', methods=[measurement('SDF-forced', .1, ('m00',))])
        rows.append(r)
        folder = tmp_path/'w4'/r['case']
        folder.mkdir(parents=True)
        np.savez(folder/'data.npz', target=np.ones((2, 12)),
                 **{'SDF-forced': np.full((2, 12), 1.1), 'SDF-contour': np.full((2, 12), 1.05)})
    summary = dict(channels=list(CHANNELS), dpi=30, records=rows, specialized=[], representation=[])
    source = tmp_path/'summary.json'
    source.write_text(json.dumps(summary))
    evidence = [source, *tmp_path.glob('w4/*/data.npz')]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    report = generate(tmp_path, max_cases=2)
    text = report.read_text()
    assert 'poor near corners/sheets' in text
    assert 'local-model + evaluation residual' in text
    assert 'cases 3–3' in text
    assert 'overview.png' not in text
    for link in re.findall(r'!\[[^]]*\]\(([^)]+)\)', text):
        assert (tmp_path/link).is_file(), link
    manifest = json.loads((tmp_path/'report-data.json').read_text())
    assert len(manifest['plots']) == 6
    assert manifest['schemes'][0]['cases'] == [f'future-shape-{i}' for i in range(3)]
    assert set(manifest['schemes'][0]['budgets']) == {'geometry', 'local + evaluation'}
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}


def test_browser_companion_and_direct_links_survive_nonembedding_markdown(tmp_path):
    from html.parser import HTMLParser
    from warpSPHBoundaries.validation.report_view import accessible_report
    path = tmp_path/'report.md'
    path.write_text('# Audit\n\nA measured result.\n\n![Moment error](figure.png)\n\n| Case | Error |\n|---|---|\n| tiny | 0.1 |\n')
    companion = accessible_report(path)
    first = path.read_text()
    accessible_report(path)
    assert path.read_text() == first
    assert '[Open figure — Moment error](figure.png)' in first
    assert '![Moment error](figure.png)' in first
    html = companion.read_text()
    assert '<table>' in html and '<img src="figure.png" alt="Moment error"' in html
    assert 'A measured result.' in html


def test_stress_report_uses_executed_guarded_images_and_archives_old_mask(tmp_path, monkeypatch):
    from warpSPHBoundaries.validation.audit_report import _stress_evidence, refresh_stress_sections
    from warpSPHBoundaries.validation import sdf_stress
    root = tmp_path/'figures'
    stress = root/'boundary-atlas'/'sdf-stress'
    folder = stress/'sharp-corner'
    folder.mkdir(parents=True)
    for name in ('moments.png', 'gradient-moments.png'):
        (folder/name).write_bytes(b'preserved forced diagnostic')
    # Old metadata deliberately has no current policy identifier.
    (stress/'summary.json').write_text(json.dumps(dict(cases=[dict(name='sharp-corner', assumption='Corner', reference='closed form')])))
    checkpoint = root/'sdf-guard-cpu'
    checkpoint.mkdir()
    (checkpoint/'checkpoint.json').write_text(json.dumps(dict(policy='sampled-cell-interval-v1', device='cpu')))
    grid = np.linspace(-1, 1, 3)
    values = np.ones((9, 12))*.1
    guarded = values.copy()
    guarded[:, 3:6] = np.nan
    np.savez(checkpoint/'corner.npz', grid=grid, target=values, guarded=guarded)
    report_dir = root/'audit'
    report_dir.mkdir()
    path = report_dir/'report.md'
    path.write_text('# Audit\n\nPreserved introduction.\n\n## Local-model stress geometries\n\n![stale](old.png)\n\n## Limits\n\nPreserved limits.\n')
    evidence = [stress/'summary.json', checkpoint/'corner.npz', folder/'moments.png']
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    # Publishing from raw cached production output must not evaluate a model.
    monkeypatch.setattr(sdf_stress, 'sampled_model', lambda *a, **k: pytest.fail('report reran evaluator'))
    assert refresh_stress_sections(root) == [path]
    text = path.read_text()
    images = re.findall(r'!\[[^]]*\]\(([^)]+)\)', text)
    assert len(images) == 2 and all('guarded-v1-' in p for p in images)
    assert 'Current production result: guarded Scene' in text
    assert 'pre-guard policy mask' in text and 'not be read as current acceptance' in text
    assert text.index('guarded-v1-moments.png') < text.index('Forced-model diagnostic archive')
    assert 'Preserved introduction.' in text and 'Preserved limits.' in text
    assert 'stale' not in text
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    assert all((report_dir/p).exists() for p in images)
    assert 'guarded-v1-moments.png' in path.with_suffix('.html').read_text()
    refresh_stress_sections(root)
    assert path.read_text() == text
    # Future generator calls use the same current-first/diagnostic-link policy.
    lines = []
    _stress_evidence(lines, report_dir)
    assert '![sharp-corner CURRENT guarded' in '\n'.join(lines)


def test_stress_without_current_checkpoint_never_embeds_forced_as_production(tmp_path):
    from warpSPHBoundaries.validation.audit_report import _stress_evidence
    stress = tmp_path/'boundary-atlas'/'sdf-stress'
    folder = stress/'sharp-corner'
    folder.mkdir(parents=True)
    (stress/'summary.json').write_text(json.dumps(dict(cases=[dict(name='sharp-corner', assumption='Corner', reference='closed form')])))
    (folder/'moments.png').touch()
    lines = []
    _stress_evidence(lines, tmp_path/'report')
    text = '\n'.join(lines)
    assert '![' not in text
    assert 'No current executed guarded spatial checkpoint' in text
    assert '[Historical forced diagnostic' in text


def test_distance_only_budget_has_separate_geometry_and_reconstruction_terms():
    target = np.ones((2, 12))
    numerical = target*.001
    reconstruction = target*.02
    geometry = target*.3
    name = 'SDF-distance-only'
    arrays = dict(target=target, **{name: target+numerical+reconstruction+geometry,
                  name+'_numerical_error': numerical, name+'_sampling_error': reconstruction,
                  name+'_geometry_error': geometry})
    budget = signed_budget(arrays, name)
    assert set(budget) == {'total', 'numerical', 'SDF reconstruction', 'half-plane geometry'}
    np.testing.assert_allclose(budget['total'], budget['numerical']+budget['SDF reconstruction']+budget['half-plane geometry'])
    assert 'no curvature pole' in scheme_info(name)[0]
