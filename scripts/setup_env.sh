#!/usr/bin/env bash
# =============================================================================
# setup_env.sh -- one-shot CANFAR setup for Pilot Proxy archive runs.
#
# Builds a clean Python venv, installs Pilot Proxy, and -- on a GPU node
# -- installs CuPy (a pip wheel matched to the node's CUDA) and compiles/stages
# the CUDA kernel for the visible GPU. Idempotent: re-running rebuilds the venv
# from scratch (python -m venv --clear).
#
#   bash scripts/setup_env.sh
#
# Override defaults via environment variables:
#   VENV_DIR=~/envs/pilot-proxy PYTHON=python3.12 \
#     PILOT_PROXY_DIR=~/src/pilot-proxy \
#     bash scripts/setup_env.sh
# Set PILOT_PROXY_SKIP_REGISTRY=1 when no remote GPU session will be launched.
# Set PILOT_PROXY_USE_SYSTEM_PACKAGES=0 for an isolated local environment.
# To deliberately adopt and clear a genuine pre-guard Python venv, also set:
#   PILOT_PROXY_ADOPT_LEGACY_VENV=1
# =============================================================================
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/pilot-proxy-venv}"
PILOT_PROXY_ADOPT_LEGACY_VENV="${PILOT_PROXY_ADOPT_LEGACY_VENV:-0}"
PILOT_PROXY_SKIP_REGISTRY="${PILOT_PROXY_SKIP_REGISTRY:-0}"
PILOT_PROXY_USE_SYSTEM_PACKAGES="${PILOT_PROXY_USE_SYSTEM_PACKAGES:-1}"
PYTHON="${PYTHON:-python3.12}"
PILOT_PROXY_DIR="${PILOT_PROXY_DIR:-$HOME/pilot-proxy}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_GUARD="${SCRIPT_DIR}/setup_env_guard.py"

case "${PILOT_PROXY_SKIP_REGISTRY}" in
    0|1) ;;
    *)
        echo "ERROR: PILOT_PROXY_SKIP_REGISTRY must be 0 or 1." >&2
        exit 1
        ;;
esac
case "${PILOT_PROXY_USE_SYSTEM_PACKAGES}" in
    0|1) ;;
    *)
        echo "ERROR: PILOT_PROXY_USE_SYSTEM_PACKAGES must be 0 or 1." >&2
        exit 1
        ;;
esac

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} is not on PATH (set PYTHON=... to override)." >&2
    exit 1
fi

for d in "${PILOT_PROXY_DIR}"; do
    if [[ ! -d "${d}" ]]; then
        echo "ERROR: directory does not exist: ${d}" >&2
        exit 1
    fi
    if [[ ! -f "${d}/pyproject.toml" ]]; then
        echo "ERROR: ${d} does not look like a Python checkout (no pyproject.toml)." >&2
        exit 1
    fi
done

# --- venv --------------------------------------------------------------------
# The next command clears every entry under VENV_DIR. Resolve symlinks, refuse
# protected/foreign directories, create the target root, and durably bind a
# sidecar to that directory identity. The sidecar survives --clear, so an
# interrupted rebuild is safely retryable, but not after deletion/recreation.
# Validated legacy Python venvs require the explicit adoption opt-in.
_venv_adoption_args=()
case "${PILOT_PROXY_ADOPT_LEGACY_VENV}" in
    0) ;;
    1) _venv_adoption_args=(--adopt-legacy-venv) ;;
    *)
        echo "ERROR: PILOT_PROXY_ADOPT_LEGACY_VENV must be 0 or 1." >&2
        exit 1
        ;;
esac
VENV_DIR="$(
    "${PYTHON}" "${VENV_GUARD}" check \
        --venv "${VENV_DIR}" \
        --home "${HOME}" \
        --checkout "${PILOT_PROXY_DIR}" \
        --checkout "${SCRIPT_DIR}/.." \
        "${_venv_adoption_args[@]}"
)"
unset _venv_adoption_args
echo "==> (re)creating venv at '${VENV_DIR}' with ${PYTHON}"
# Session images can expose their CuPy stack through system packages. Local
# workstations can set PILOT_PROXY_USE_SYSTEM_PACKAGES=0 for a clean venv.
_venv_create_args=(--clear)
if [[ "${PILOT_PROXY_USE_SYSTEM_PACKAGES}" == "1" ]]; then
    _venv_create_args+=(--system-site-packages)
fi
"${PYTHON}" -m venv "${_venv_create_args[@]}" "${VENV_DIR}"
unset _venv_create_args
# A successfully created environment also receives a human-visible marker
# inside it. Rerun authorization comes from the durable sidecar written above.
"${PYTHON}" "${VENV_GUARD}" mark --venv "${VENV_DIR}"
# Keep ~/.local user-site packages from leaking into the venv (persisted on the
# activate script so every future shell that sources it stays isolated too).
echo 'export PYTHONNOUSERSITE=1' >> "${VENV_DIR}/bin/activate"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1

# Keep temporary files on the Linux filesystem.
PILOT_PROXY_TMPDIR="${PILOT_PROXY_TMPDIR:-/tmp}"
if [[ ! -d "${PILOT_PROXY_TMPDIR}" || ! -w "${PILOT_PROXY_TMPDIR}" ]]; then
    echo "ERROR: temporary directory is not writable: ${PILOT_PROXY_TMPDIR}" >&2
    exit 1
fi
export TMPDIR="${PILOT_PROXY_TMPDIR}"
printf 'export TMPDIR=%q\n' "${PILOT_PROXY_TMPDIR}" >> "${VENV_DIR}/bin/activate"

# --- CANFAR Harbor registry credentials --------------------------------------
# launch_gpu_session.py (skaha) needs a Harbor CLI secret to pull the session
# image. Capture it once, store it chmod 600, and source it from the venv
# activate so every future shell has it. Prompt only when interactive and the
# creds were not already provided (via env var or a previously saved file).
CANFAR_ENV_FILE="${CANFAR_ENV_FILE:-$HOME/.canfar_registry.env}"
if [[ "${PILOT_PROXY_SKIP_REGISTRY}" == "1" ]]; then
    echo "==> skipping Harbor registry setup"
elif [[ -n "${CANFAR_REGISTRY_USER:-}" && -n "${CANFAR_REGISTRY_SECRET:-}" ]]; then
    ( umask 077; printf 'export CANFAR_REGISTRY_USER=%q\nexport CANFAR_REGISTRY_SECRET=%q\n' \
        "${CANFAR_REGISTRY_USER}" "${CANFAR_REGISTRY_SECRET}" > "${CANFAR_ENV_FILE}" )
    chmod 600 "${CANFAR_ENV_FILE}"
    echo "==> CANFAR credentials taken from the environment, saved to ${CANFAR_ENV_FILE}"
elif [[ -f "${CANFAR_ENV_FILE}" ]]; then
    echo "==> reusing CANFAR credentials from ${CANFAR_ENV_FILE}"
elif [[ -t 0 ]]; then
    _canfar_default_user="${USER:-$(id -un)}"
    echo
    echo "CANFAR Harbor registry login (needed to launch GPU sessions)."
    echo "  Get your CLI secret at https://images.canfar.net -> (your profile) -> CLI secret."
    read -r    -p "  CANFAR_REGISTRY_USER [${_canfar_default_user}]: " _canfar_u
    read -r -s -p "  CANFAR_REGISTRY_SECRET (hidden): " _canfar_s
    echo
    if [[ -n "${_canfar_s}" ]]; then
        ( umask 077; printf 'export CANFAR_REGISTRY_USER=%q\nexport CANFAR_REGISTRY_SECRET=%q\n' \
            "${_canfar_u:-${_canfar_default_user}}" "${_canfar_s}" > "${CANFAR_ENV_FILE}" )
        chmod 600 "${CANFAR_ENV_FILE}"
        echo "==> CANFAR credentials saved to ${CANFAR_ENV_FILE} (chmod 600)"
    else
        echo "==> no secret entered -- skipping (set creds before launch_gpu_session.py)"
    fi
    unset _canfar_u _canfar_s _canfar_default_user
else
    echo "==> CANFAR_REGISTRY_USER/SECRET unset and no TTY -- skipping credential prompt."
    echo "    Set them before launch_gpu_session.py (https://images.canfar.net -> profile -> CLI secret)."
fi
# Source the saved creds from the venv activate (idempotent) and this shell.
if [[ "${PILOT_PROXY_SKIP_REGISTRY}" != "1" && -f "${CANFAR_ENV_FILE}" ]]; then
    if ! grep -qF "${CANFAR_ENV_FILE}" "${VENV_DIR}/bin/activate"; then
        echo "[ -f \"${CANFAR_ENV_FILE}\" ] && . \"${CANFAR_ENV_FILE}\"" >> "${VENV_DIR}/bin/activate"
    fi
    # shellcheck disable=SC1090
    . "${CANFAR_ENV_FILE}"
fi

python -m pip install -U pip setuptools wheel packaging

# --- scientific stack + editable installs -----------------------------------
echo "==> installing scientific stack"
python -m pip install -U numpy scipy h5py matplotlib pytest

echo "==> installing the tested Datatrail revision"
python -m pip install -r "${PILOT_PROXY_DIR}/requirements/archive.txt"

echo "==> installing pilot-proxy (editable, + archive/test extras)"
python -m pip install -e "${PILOT_PROXY_DIR}[archive,test]"

# Survey shells out to Datatrail, which must resolve inside this venv.
if [[ "$(command -v datatrail || true)" != "${VIRTUAL_ENV}/bin/datatrail" ]]; then
    echo "ERROR: datatrail is not resolving inside the venv." >&2
    exit 1
fi

# --- canfar client (Science Platform / skaha; for launch_gpu_session.py) -----
echo "==> installing canfar (skaha client used by launch_gpu_session.py)"
python -m pip install canfar \
    || echo "WARNING: 'pip install canfar' failed; launch_gpu_session.py will not work until it is installed."

# --- CuPy: resolve through Pilot Proxy's archive runtime ----------------------
# The scan resolver prefers the CuPy the
# session image ships, and on a GPU node with no image CuPy installs the
# matching wheel.
echo "==> CuPy"
python - <<'PYEOF' || true
import shutil
from pilot_proxy.archive import accel
cp = accel.import_cupy()
if cp is not None:
    print(f"    CuPy {cp.__version__} already importable (from the session image)")
elif shutil.which("nvidia-smi"):
    try:
        cp = accel.ensure_cupy(install=True)
        print(f"    installed CuPy {getattr(cp, '__version__', '?')}")
    except Exception as exc:
        print(f"    CuPy not installed: {exc}")
else:
    print("    no GPU here -- skipping (CuPy is only needed on the GPU detector node)")
PYEOF

# --- CuPy CUDA headers (so the runtime JIT can build kernels) -----------------
# The runtime wheel ('cupy-cudaXXx' -- shipped by the image or installed by accel
# above) provides CUDA *libraries* but not the toolkit *headers* CuPy's nvrtc needs
# to compile its elementwise/reduction kernels on first use. cuda-pathfinder finds
# those in the 'nvidia-cuda-*' header wheels that the '[ctk]' extra pulls in. (The
# image's bundled nvcc builds our kernel fine but keeps no usable headers under its
# include/, so CuPy cannot borrow them.) Probe a real JIT compile and add the header
# wheels only when it fails -- an image that already ships headers then pays nothing.
if command -v nvidia-smi >/dev/null 2>&1; then
    _cupy_jit_probe() {
        python - <<'PY' >/dev/null 2>&1
import cupy as cp
a = cp.arange(8, dtype=cp.float32)
assert int((a * a).sum()) == 140
PY
    }
    if _cupy_jit_probe; then
        echo "==> CuPy JIT: CUDA headers already available"
    else
        pkg="$(python - <<'PY' 2>/dev/null
from pilot_proxy.archive import accel
print(accel.cupy_package(accel.detect_cuda_major() or 12))
PY
)"
        [[ -n "${pkg}" ]] || pkg="cupy-cuda12x"
        echo "==> CuPy JIT: CUDA headers missing -- installing toolkit header wheels (${pkg}[ctk])"
        python -m pip install "${pkg}[ctk]"
        if _cupy_jit_probe; then
            echo "    CuPy JIT: ok after ${pkg}[ctk]"
        else
            echo "ERROR: CuPy still cannot JIT-compile after installing ${pkg}[ctk]." >&2
            echo "       The detector compiles CUDA kernels at runtime; aborting setup." >&2
            exit 1
        fi
    fi
    unset -f _cupy_jit_probe
fi

# --- CUDA kernel -------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi found, but nvcc is not on PATH." >&2
        echo "       Load the CUDA toolkit/module or set PATH/NVCC before rerunning." >&2
        exit 1
    fi
    SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
    if [[ -z "${SM}" ]]; then
        echo "ERROR: could not determine GPU compute capability from nvidia-smi." >&2
        exit 1
    fi
    echo "==> building and staging CUDA kernel for detected arch sm_${SM}"
    # cuda/Makefile keys the build on a config stamp (arch + kernel flags), so
    # a rerun on a different GPU type recompiles automatically -- no need to
    # delete cuda/libfstatistic.so first. Still drop the staged cache copy so a
    # failed build fails loud rather than silently serving a stale kernel.
    make -C "${PILOT_PROXY_DIR}" clean-cache
    make -C "${PILOT_PROXY_DIR}" build-kernel SM="${SM}"
else
    echo "==> no GPU detected (nvidia-smi absent) -- skipping kernel build"
    echo "    the detector needs a GPU node; a CPU-only host can still run the Python tests"
fi

# --- sanity checks -----------------------------------------------------------
echo "==> verifying Python packages"
python - <<'PY'
import importlib.util
import sys
print("    python", sys.version.split()[0], sys.executable)
for name in ["numpy", "scipy", "h5py", "matplotlib", "pytest", "dtcli", "pilot_proxy"]:
    ok = importlib.util.find_spec(name) is not None
    print(f"    {name:12s}: {'OK' if ok else 'MISSING'}")
    if not ok:
        raise SystemExit(1)
if importlib.util.find_spec("cupy"):
    try:
        import cupy
        print("    cupy       :", cupy.__version__, "CUDA-rt", cupy.cuda.runtime.runtimeGetVersion())
    except Exception as exc:                       # importable but no usable runtime here
        print("    cupy       : present, runtime not usable here:", type(exc).__name__)
else:
    print("    cupy       : MISSING (no image CuPy and none installed)")
print("    canfar     :", "OK" if importlib.util.find_spec("canfar") else "MISSING (needed by launch_gpu_session.py)")
PY

echo "==> verifying archive components"
python - <<'PY'
from pilot_proxy.archive.sources import CadcDatatrailSource, LocalDirectorySource
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader

print("    sources :", LocalDirectorySource.info.name, CadcDatatrailSource.info.name)
print("    reader  :", ChimeBasebandPackedReader.info.name)
print("    analyzer:", PilotProxyDetectorAnalyzer.info.name)
PY
echo "    survey CLI : datatrail -> $(command -v datatrail || echo 'NOT FOUND')"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> verifying PilotProxy CUDA kernel load"
    (cd "${PILOT_PROXY_DIR}" && PYTHONPATH=src python - <<'PY'
from pilot_proxy.kernel import FStatKernel
kernel = FStatKernel()
print("    specs   :", kernel.specs.as_descriptive_dict())
print("    features:", kernel.features.as_dict())
print("    version :", kernel.version.as_string())
PY
    )
fi

echo "==> running offline archive integration tests"
(cd "${PILOT_PROXY_DIR}" && PYTHONPATH=src python -m pytest tests/archive -q)

echo
echo "==> done. In a new shell, activate with:"
echo "      source ${VENV_DIR}/bin/activate"
