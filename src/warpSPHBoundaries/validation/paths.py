"""Keep regenerable development evidence outside the source checkout.

WARPSPH_DEVELOPMENT_ROOT overrides the project archive directory. Study
--output/--baseline options remain available for individual runs.
"""
import os
from pathlib import Path


def development_root():
    configured = os.environ.get('WARPSPH_DEVELOPMENT_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    # Editable/source installs: src/warpSPHBoundaries/validation/paths.py.
    checkout = Path(__file__).resolve().parents[3]
    if (checkout/'pyproject.toml').exists():
        return checkout.parent/'warpSPHDevelopment'/'curvatureBoundaries'
    # Installed wheels have no checkout; retain an explicit, documented home
    # location rather than treating site-packages as a development directory.
    return Path.home()/'dev'/'warpSPHDevelopment'/'curvatureBoundaries'


def evidence_path(name=''):
    return development_root()/'results'/'figures'/name
