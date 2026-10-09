#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/publish_pypi.sh [--build-only] [--testpypi] [--skip-existing]

Builds the package, runs twine checks, and uploads it to PyPI or TestPyPI.
--build-only builds and checks dist/* and stops before any upload.

Credential sources:
  1. ~/.pypirc configured with scripts/setup_pypi_token.sh
  2. PYPI_TOKEN or TEST_PYPI_TOKEN environment variables
EOF
}

repository="pypi"
repository_url="https://upload.pypi.org/legacy/"
skip_existing=0
build_only=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --testpypi)
            repository="testpypi"
            repository_url="https://test.pypi.org/legacy/"
            shift
            ;;
        --skip-existing)
            skip_existing=1
            shift
            ;;
        --build-only)
            build_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Could not find Python executable '$python_bin'. Set PYTHON_BIN to override." >&2
    exit 1
fi

"$python_bin" - <<'PY'
from pathlib import Path
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:
    print("Python 3.11+ is required for this release script because it uses tomllib.", file=sys.stderr)
    sys.exit(1)

pyproject_version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
init_text = Path("src/warpSPHBoundaries/__init__.py").read_text()
match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, re.MULTILINE)
if match is None:
    print("Could not find __version__ in src/warpSPHBoundaries/__init__.py", file=sys.stderr)
    sys.exit(1)

package_version = match.group(1)
if pyproject_version != package_version:
    print(
        f"Version mismatch: pyproject.toml has {pyproject_version}, "
        f"but src/warpSPHBoundaries/__init__.py has {package_version}",
        file=sys.stderr,
    )
    sys.exit(1)

print(pyproject_version)
PY

if ! "$python_bin" -c 'import build, twine' >/dev/null 2>&1; then
    echo "Missing build dependencies. Install them with: $python_bin -m pip install build twine" >&2
    exit 1
fi

if [[ -d .git ]]; then
    if [[ -n "$(git status --short)" ]]; then
        echo "Warning: git working tree is not clean. Publishing will use the current local contents." >&2
    fi
fi

rm -rf build dist
"$python_bin" -m build
"$python_bin" -m twine check dist/*

if [[ $build_only -eq 1 ]]; then
    echo "Built and checked (no upload):"
    ls -la dist/
    exit 0
fi

upload_args=("$python_bin" -m twine upload)
if [[ $skip_existing -eq 1 ]]; then
    upload_args+=(--skip-existing)
fi

if [[ "$repository" == "pypi" ]]; then
    if [[ -n "${PYPI_TOKEN:-}" ]]; then
        TWINE_USERNAME="__token__" TWINE_PASSWORD="$PYPI_TOKEN" "${upload_args[@]}" --repository-url "$repository_url" dist/*
    else
        "${upload_args[@]}" --repository pypi dist/*
    fi
else
    if [[ -n "${TEST_PYPI_TOKEN:-}" ]]; then
        TWINE_USERNAME="__token__" TWINE_PASSWORD="$TEST_PYPI_TOKEN" "${upload_args[@]}" --repository-url "$repository_url" dist/*
    else
        "${upload_args[@]}" --repository testpypi dist/*
    fi
fi