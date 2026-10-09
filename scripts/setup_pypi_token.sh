#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_pypi_token.sh [pypi|testpypi]

Stores a PyPI API token in ~/.pypirc with file mode 600.

Environment variables:
  PYPI_TOKEN        Token used when the target is pypi
  TEST_PYPI_TOKEN   Token used when the target is testpypi
EOF
}

target="${1:-pypi}"

case "$target" in
    pypi)
        repository_url="https://upload.pypi.org/legacy/"
        token_var="PYPI_TOKEN"
        ;;
    testpypi)
        repository_url="https://test.pypi.org/legacy/"
        token_var="TEST_PYPI_TOKEN"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unsupported target: $target" >&2
        usage >&2
        exit 1
        ;;
esac

token="${!token_var:-}"
if [[ -z "$token" ]]; then
    read -r -s -p "Enter ${target} API token: " token
    echo
fi

if [[ -z "$token" ]]; then
    echo "No token provided." >&2
    exit 1
fi

if [[ "$token" != pypi-* ]]; then
    echo "The token does not look like a PyPI API token. Expected a value starting with 'pypi-'." >&2
    exit 1
fi

pypirc_path="$HOME/.pypirc"
backup_path="$HOME/.pypirc.bak"
temp_file="$(mktemp)"
cleanup() {
    rm -f "$temp_file"
}
trap cleanup EXIT

if [[ -f "$pypirc_path" ]]; then
    cp "$pypirc_path" "$backup_path"
    awk -v target_section="[$target]" '
        BEGIN { skip = 0 }
        /^\[distutils\]$/ { skip = 1; next }
        /^\[/ {
            if (skip == 1) {
                skip = 0
            }
            if ($0 == target_section) {
                skip = 1
                next
            }
        }
        skip == 0 { print }
    ' "$pypirc_path" > "$temp_file"
else
    : > "$temp_file"
fi

umask 077
{
    cat <<EOF
[distutils]
index-servers =
    pypi
    testpypi

EOF
    if [[ -s "$temp_file" ]]; then
        cat "$temp_file"
        printf '\n'
    fi
    cat <<EOF
[$target]
repository = $repository_url
username = __token__
password = $token
EOF
} > "$pypirc_path"

chmod 600 "$pypirc_path"

echo "Wrote credentials for '$target' to $pypirc_path"
if [[ -f "$backup_path" ]]; then
    echo "Previous config backed up to $backup_path"
fi