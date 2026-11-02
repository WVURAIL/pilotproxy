#!/usr/bin/env bash
set -euo pipefail

# Standalone CUDA and GNU Radio release check. It runs in the virtual
# environment of the repository (README: Installation and checks). The defaults
# use the active interpreter and the GPU architecture detected by cuda/Makefile,
# and these variables override them:
#   SM                    GPU compute capability (default: auto-detected by cuda/Makefile)
#   CUDA_PYTHON           interpreter with cupy and this package installed (default: active python3)
#   GNURADIO_PYTHON       interpreter with GNU Radio (for the ATSC generator)
#   CUDA_LD_LIBRARY_PATH  extra CUDA/driver lib path if needed (WSL: /usr/lib/wsl/lib)
# The script needs a CUDA-capable GPU. The CPU-only checks are in the README.
SM="${SM:-}"
CUDA_PYTHON="${CUDA_PYTHON:-python3}"
GNURADIO_PYTHON="${GNURADIO_PYTHON:-/usr/bin/python3}"
CUDA_LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH

if [ -n "$SM" ]; then
    make build-kernel SM="$SM"
else
    make build-kernel
fi

PYTHONNOUSERSITE=1 PYTHONPATH=src "$GNURADIO_PYTHON" \
  -m pilot_proxy.testbench.generate_atsc_signal \
  --output-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --num-iq-samples 600000

PYTHONPATH=src "$CUDA_PYTHON" \
  -m pilot_proxy.testbench.audit_atsc_signal \
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile

PYTHONPATH=src "$CUDA_PYTHON" \
  -m pilot_proxy.testbench.quantize \
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --physical-channel 14 \
  --frame-size-samples 16384 \
  --num-input-streams 1

LD_LIBRARY_PATH="$CUDA_LD_LIBRARY_PATH" PYTHONPATH=src "$CUDA_PYTHON" \
  -m pilot_proxy.testbench.evaluate_snr \
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --output-dir generated/quickstart/dtv_snr_eval \
  --physical-channel 14 \
  --frame-size-samples 16384 \
  --num-input-streams 1 \
  --requested-data-shelf-snr-db -26 \
  --noise-trials 10 \
  --noise-source gnuradio \
  --gnuradio-python "$GNURADIO_PYTHON"

# pilot identity comes from the metadata.json sidecar quantize wrote above
LD_LIBRARY_PATH="$CUDA_LD_LIBRARY_PATH" PYTHONPATH=src "$CUDA_PYTHON" \
  -m pilot_proxy.detect \
  --input-detector-matrix generated/detector_input/detector_matrix_i4.npy
