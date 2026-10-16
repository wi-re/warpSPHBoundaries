"""Development evidence must not default to the source checkout."""
from pathlib import Path

from warpSPHBoundaries.validation.paths import development_root, evidence_path


def test_default_evidence_is_a_sibling_project_archive(monkeypatch):
    monkeypatch.delenv('WARPSPH_DEVELOPMENT_ROOT', raising=False)
    checkout = Path(__file__).resolve().parents[2]
    expected = checkout.parent/'warpSPHDevelopment'/'curvatureBoundaries'
    assert development_root() == expected
    assert evidence_path('sdf-accuracy-cpu') == expected/'results/figures/sdf-accuracy-cpu'
    assert not evidence_path().is_relative_to(checkout)


def test_archive_root_can_be_overridden_without_writing(monkeypatch, tmp_path):
    root = tmp_path/'development'
    monkeypatch.setenv('WARPSPH_DEVELOPMENT_ROOT', str(root))
    assert development_root() == root
    assert evidence_path('boundary-atlas/sdf-stress') == root/'results/figures/boundary-atlas/sdf-stress'
    assert not root.exists()
