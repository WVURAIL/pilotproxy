#!/usr/bin/env bash
# Fail when local build/review artifacts that should not be committed exist.
set -euo pipefail

blocked_paths=(
  "generated"
  "docs/out"
  ".pytest_cache"
  ".ruff_cache"
  ".idea"
  "inspection"
  "src/pilot_proxy.egg-info"
  "cuda/libfstatistic.so"
  "cuda/test_fstat_reference"
  "cuda/test_c_header.o"
)

found=()
for path in "${blocked_paths[@]}"; do
  if [[ -e "$path" ]]; then
    found+=("$path")
  fi
done

while IFS= read -r -d '' path; do
  found+=("$path")
done < <(find . -type d -name __pycache__ -print0)

if [[ ${#found[@]} -gt 0 ]]; then
  printf 'Commit hygiene check failed. Remove local artifacts before commit:\n' >&2
  printf '  %s\n' "${found[@]}" >&2
  exit 1
fi

echo "Commit hygiene check passed."
