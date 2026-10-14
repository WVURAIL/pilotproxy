#!/usr/bin/env bash
#
#   run_estimator_transfer.sh [RESULT_DIR]
#   run_estimator_transfer.sh --check-runtime
#
# The second form resolves and reports the Python that would be used and exits
# without running anything, which is the cheap way to find out whether this
# machine can drive the sweep before committing hours to it.
#
# Runtime selection, in order: $PILOT_PROXY_PYTHON, an activated virtualenv,
# the repository's own .venv, the venv scripts/setup_env.sh builds
# ($VENV_DIR, default ~/pilot-proxy-venv), then python3/python on PATH.
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
check_runtime_only=0
if [[ ${1:-} == "--check-runtime" ]]; then
  check_runtime_only=1
  shift
fi
result_dir=${1:-"$project_dir/../estimator_transfer_2026-08-25"}
max_parallel=${ESTIMATOR_TRANSFER_JOBS:-4}

# Every point below is launched as "$runtime", so a runtime that does not
# resolve fails 26 times over inside per-point logs with nothing on the
# console --- which is exactly how the hardcoded ~/rail/venvs/ppci default
# rotted unnoticed once that venv was removed. Discover the interpreter from
# the conventions this repository already documents, and prove the one we pick
# can import what the sweep needs before spending the first CPU-hour on it.
# The probe runs under the same PYTHONNOUSERSITE=1 PYTHONPATH=src environment
# as the sweep points themselves, so an interpreter whose cupy lives only in
# ~/.local is rejected here rather than at the first point. It must echo the
# sentinel: a non-Python executable would otherwise read the here-document,
# ignore it and exit 0.
probe_runtime() {
  local reply
  reply=$(PYTHONNOUSERSITE=1 PYTHONPATH="$project_dir/src" "$1" - <<'PROBE' 2>/dev/null
import cupy  # both backends below are --detector/synthesis-backend cuda
from pilot_proxy.testbench import evaluate_snr  # noqa: F401
print("estimator-transfer-runtime-ok")
PROBE
  ) || return 1
  [[ $reply == "estimator-transfer-runtime-ok" ]]
}

resolve_runtime() {
  local candidates=() rejected=() cand found
  if [[ -n "${PILOT_PROXY_PYTHON:-}" ]]; then
    # An explicit override is a deliberate choice: use it or say why not.
    candidates=("$PILOT_PROXY_PYTHON")
  else
    [[ -n "${VIRTUAL_ENV:-}" ]] && candidates+=("$VIRTUAL_ENV/bin/python")
    candidates+=("$project_dir/.venv/bin/python")
    candidates+=("${VENV_DIR:-$HOME/pilot-proxy-venv}/bin/python")
    for name in python3 python; do
      if found=$(command -v "$name" 2>/dev/null); then
        candidates+=("$found")
      fi
    done
  fi

  for cand in "${candidates[@]}"; do
    if [[ -x "$cand" ]] && probe_runtime "$cand"; then
      printf '%s\n' "$cand"
      return 0
    fi
    rejected+=("$cand")
  done

  {
    echo "No Python able to run the estimator-transfer sweep was found."
    echo "Each candidate must be executable and must import cupy and"
    echo "pilot_proxy.testbench.evaluate_snr with PYTHONPATH=$project_dir/src."
    echo "Tried, in order:"
    printf '  %s\n' "${rejected[@]}"
    echo "Set PILOT_PROXY_PYTHON to the interpreter you want used."
  } >&2
  return 2
}

runtime=$(resolve_runtime)
echo "estimator transfer: runtime $runtime ($("$runtime" -V 2>&1))"
if (( check_runtime_only )); then
  exit 0
fi

if [[ ! "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
  echo "ESTIMATOR_TRANSFER_JOBS must be a positive integer." >&2
  exit 2
fi

if [[ -e "$result_dir" ]]; then
  echo "Result directory already exists: $result_dir" >&2
  exit 2
fi

mkdir -p "$result_dir"
cd "$project_dir"

common=(
  -m pilot_proxy.testbench.evaluate_snr
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile
  --waveform-audit-json generated/atsc/atsc_waveform_audit.json
  --physical-channel 14
  --frame-size-samples 16384
  --num-input-streams 4
  --detector-backend cuda
  --synthesis-backend cuda
  --noise-source gnuradio
  --gnuradio-python /usr/bin/python3
  --pilot-below-data-db-from-audit
  --spectral-sense normal
  --no-reference-archive-phase
)

run_point() {
  local label=$1
  local snr_db=$2
  local trials=$3
  local seed=$4
  local output_dir=$5

  mkdir -p "$(dirname "$output_dir")"
  PYTHONNOUSERSITE=1 PYTHONPATH=src "$runtime" -u "${common[@]}" \
    --requested-data-shelf-snr-db "$snr_db" \
    --noise-trials "$trials" --seed "$seed" \
    --output-dir "$output_dir" \
    >"$result_dir/${label}.log" 2>&1
}

wait_for_jobs() {
  local status=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

point_pids=()
queue_point() {
  run_point "$@" &
  point_pids+=("$!")
  if (( ${#point_pids[@]} >= max_parallel )); then
    wait_for_jobs "${point_pids[@]}"
    point_pids=()
  fi
}

finish_points() {
  if (( ${#point_pids[@]} > 0 )); then
    wait_for_jobs "${point_pids[@]}"
    point_pids=()
  fi
}

queue_point extension_m60 -60 240 2026082700 "$result_dir/extreme_low/snr_m60"
queue_point extension_m57 -57 240 2026082701 "$result_dir/extreme_low/snr_m57"
queue_point extension_m54 -54 240 2026082702 "$result_dir/extreme_low/snr_m54"
queue_point extension_m51 -51 240 2026082703 "$result_dir/extreme_low/snr_m51"
queue_point extension_m48 -48 240 2026082704 "$result_dir/extreme_low/snr_m48"
queue_point extension_m45 -45 240 2026082705 "$result_dir/extreme_low/snr_m45"
queue_point extension_p3 3 60 2026082706 "$result_dir/high"
queue_point extension_p6 6 60 2026082707 "$result_dir/high/snr_p6"
queue_point extension_p9 9 60 2026082708 "$result_dir/high/snr_p9"
queue_point extension_p12 12 60 2026082709 "$result_dir/high/snr_p12"
queue_point extension_p15 15 60 2026082710 "$result_dir/high/snr_p15"
queue_point extension_p18 18 60 2026082711 "$result_dir/high/snr_p18"
queue_point extension_p21 21 60 2026082712 "$result_dir/high/snr_p21"
queue_point extension_p24 24 60 2026082713 "$result_dir/high/snr_p24"
queue_point extension_p27 27 60 2026082714 "$result_dir/high/snr_p27"
queue_point extension_p30 30 60 2026082715 "$result_dir/high/snr_p30"
queue_point high_p33 33 60 2026082812 "$result_dir/high/snr_p33"
queue_point high_p36 36 60 2026082813 "$result_dir/high/snr_p36"
queue_point high_p39 39 60 2026082814 "$result_dir/high/snr_p39"
queue_point high_p42 42 60 2026082815 "$result_dir/high/snr_p42"
queue_point high_p45 45 60 2026082816 "$result_dir/high/snr_p45"
queue_point high_p48 48 60 2026082817 "$result_dir/high/snr_p48"
queue_point high_p51 51 60 2026082818 "$result_dir/high/snr_p51"
queue_point high_p54 54 60 2026082819 "$result_dir/high/snr_p54"
queue_point high_p57 57 60 2026082820 "$result_dir/high/snr_p57"
queue_point high_p60 60 60 2026082821 "$result_dir/high/snr_p60"
finish_points

PYTHONNOUSERSITE=1 PYTHONPATH=src "$runtime" -u "${common[@]}" \
  --snr-start-db -42 --snr-stop-db -30 --snr-step-db 3 \
  --noise-trials 240 --seed 20260825 \
  --output-dir "$result_dir/lower" \
  2>&1 | tee "$result_dir/lower.log"

PYTHONNOUSERSITE=1 PYTHONPATH=src "$runtime" -u "${common[@]}" \
  --snr-start-db -27 --snr-stop-db 0 --snr-step-db 3 \
  --noise-trials 60 --seed 20260826 \
  --output-dir "$result_dir/upper" \
  2>&1 | tee "$result_dir/upper.log"

queue_point lower_m45_part1 -45 380 2026082800 "$result_dir/lower_additional/snr_m45/part_1"
queue_point lower_m45_part2 -45 380 2026082801 "$result_dir/lower_additional/snr_m45/part_2"
queue_point lower_m42_part1 -42 380 2026082802 "$result_dir/lower_additional/snr_m42/part_1"
queue_point lower_m42_part2 -42 380 2026082803 "$result_dir/lower_additional/snr_m42/part_2"
queue_point lower_m39_part1 -39 380 2026082804 "$result_dir/lower_additional/snr_m39/part_1"
queue_point lower_m39_part2 -39 380 2026082805 "$result_dir/lower_additional/snr_m39/part_2"
queue_point lower_m36_part1 -36 380 2026082806 "$result_dir/lower_additional/snr_m36/part_1"
queue_point lower_m36_part2 -36 380 2026082807 "$result_dir/lower_additional/snr_m36/part_2"
queue_point lower_m33_part1 -33 380 2026082808 "$result_dir/lower_additional/snr_m33/part_1"
queue_point lower_m33_part2 -33 380 2026082809 "$result_dir/lower_additional/snr_m33/part_2"
queue_point lower_m30_part1 -30 380 2026082810 "$result_dir/lower_additional/snr_m30/part_1"
queue_point lower_m30_part2 -30 380 2026082811 "$result_dir/lower_additional/snr_m30/part_2"
finish_points

inputs=()
for snr in m60 m57 m54 m51 m48 m45; do
  inputs+=("$result_dir/extreme_low/snr_${snr}")
done
inputs+=("$result_dir/lower")
for snr in m45 m42 m39 m36 m33 m30; do
  inputs+=(
    "$result_dir/lower_additional/snr_${snr}/part_1"
    "$result_dir/lower_additional/snr_${snr}/part_2"
  )
done
inputs+=("$result_dir/upper" "$result_dir/high")
for snr in $(seq 6 3 60); do
  inputs+=("$result_dir/high/snr_p${snr}")
done

plot_args=()
for input_dir in "${inputs[@]}"; do
  plot_args+=(--input-csv "$input_dir/dtv_snr_summary.csv")
done
for input_dir in "${inputs[@]}"; do
  plot_args+=(--trial-csv "$input_dir/dtv_snr_eval.csv")
done

PYTHONNOUSERSITE=1 PYTHONPATH=src "$runtime" \
  -m pilot_proxy.testbench.plot_results \
  "${plot_args[@]}" \
  --conditioning-trial-csv "$result_dir/lower/dtv_snr_eval.csv" \
  --conditioning-json "$result_dir/estimator_transfer_m60_to_p60_conditioning.json" \
  --bootstrap-samples 10000 \
  --smooth-window 1 \
  --y-min-db -60 \
  --output-png "$result_dir/estimator_transfer_m60_to_p60.png" \
  --output-pdf "$result_dir/estimator_transfer_m60_to_p60.pdf" \
  --dissertation-style \
  --title "Synthetic GNU Radio input: CPU/GPU estimator transfer"
