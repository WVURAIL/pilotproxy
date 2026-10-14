#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ref="reference"
old_gap="guard"
bins="bins"
upper_ref="REFERENCE"
upper_old_gap="GUARD"
locked="LOCKED"
min_name="MIN"

old_ref_gap="${ref}_${old_gap}_${bins}"
old_upper_ref_gap="${upper_ref}_${upper_old_gap}"
old_locked_upper_ref_gap="${locked}_${upper_ref}_${upper_old_gap}"
old_min_gap="${min_name}_${upper_old_gap}_${bins}"
bare_gap_bins="\\b${old_gap}_${bins}\\b"

targets=(src tests docs README.md configs scripts cuda Makefile)
pattern="${old_ref_gap}|${old_upper_ref_gap}|${old_locked_upper_ref_gap}|${old_min_gap}|${bare_gap_bins}"

if grep -RInE \
  --exclude='*.bin' \
  --exclude-dir='out' \
  --exclude-dir='__pycache__' \
  --exclude-dir='.pytest_cache' \
  "$pattern" \
  "${targets[@]}"
then
  echo "Deprecated detector-spacing terminology found." >&2
  echo "Use skipped_guard_bins or reference_offset_bins." >&2
  exit 1
fi
