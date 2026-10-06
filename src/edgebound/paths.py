"""Repository locations in one place.

`REPO_ROOT` is the checkout (found by walking up from this file to `pyproject.toml`); it is `None` when the package is installed
without the checkout, in which case the cache of tabulated kernels goes to the user cache directory and the repo-only locations raise.
Environment: `OMNISPH_HOME` (default `~/dev/omniSPH`) is the live DFSPH reference, `EDGEBOUND_TMP` overrides the scratch directory.
"""
import os
from pathlib import Path


def _find_root():
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file() and (p / "docs").is_dir():
            return p
    return None


REPO_ROOT = _find_root()
OMNISPH_HOME = Path(os.environ.get("OMNISPH_HOME", "~/dev/omniSPH")).expanduser()


def _need_repo(sub):
    if REPO_ROOT is None:
        raise RuntimeError("edgebound.paths: %r needs the repository checkout (pyproject.toml not found above %s)" % (sub, Path(__file__).parent))
    return REPO_ROOT / sub


def results_dir():
    return _need_repo("results")


def fixtures_dir():
    return _need_repo("tests") / "fixtures"


def tmp_dir():
    """scratch (untracked): snapshots, videos, reference runs"""
    return Path(os.environ["EDGEBOUND_TMP"]) if "EDGEBOUND_TMP" in os.environ else _need_repo(".tmp")


def packaged_tables_dir():
    """the pre-tabulated tier-3 half-plane functions shipped with the package (read only when installed)"""
    return Path(__file__).resolve().parent / "data" / "tables"


def tables_dir():
    """where newly tabulated tier-3 functions are written: the packaged directory in a checkout, the user cache directory when installed"""
    if REPO_ROOT is not None:
        return packaged_tables_dir()
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "edgebound" / "tables"
