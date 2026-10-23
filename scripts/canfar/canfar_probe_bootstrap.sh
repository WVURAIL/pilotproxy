#!/usr/bin/env bash
# CANFAR probe bootstrap v2 -- pilot-proxy v5 archive run, frozen tag.
# Run inside ONE CANFAR session terminal. The session MUST use
# images.canfar.net/skaha/astroml-cuda:latest -- the non-CUDA astroml
# image has a GPU and driver but no toolkit, and cupy cannot JIT there.
#   bash /arc/home/$CANFAR_USER/pp_switch/canfar_probe_bootstrap.sh
# v2: phase 3 now restores the runtime OFFLINE from the freeze bundle's own
# wheelhouse (cut at the frozen rev), per the lock's "install only from the
# bundled wheelhouse" doctrine.  v1 tried PyPI and was correctly refused by
# hash checking (different manylinux wheel variants for the same versions).
# Read-only against the archive except ~25 timed test downloads to /tmp.
set -uo pipefail
# Site configuration: the CANFAR account name; export it before running (see README.md).
#   CANFAR_USER  the CANFAR account; the kit and the runs live under /arc/home/$CANFAR_USER/
: "${CANFAR_USER:?}"
say(){ printf '\n===== %s =====\n' "$*"; }

TAG=archive-run-source-20260901
REV=b59b5c05fed2a9509a31e206f0911e76ca2d2885
PKG=3722012957975f7d5698c24ab3bf36b59ff26dd94fd84ae75b2eb0820d8ea34a
# The kernel qualified by the cross-arch smoke (H100 / sm90). A new session on
# the same architecture reuses it instead of rebuilding, so its products carry
# the same detector_version as the running shards.
QKLIB=/arc/home/$CANFAR_USER/pp_kernels/pilotproxy-detector-core-2.3.0-sm90-33b6e1c45c47.so
QKSHA=33b6e1c45c472c65cf46031d4b009d6f6f96652b57c9bb362489f403dfeaedbd
TARSHA=f37fe0410bff5233b4a5afd99262f47d9451d446a7be9274b943f5373e079ae6
SW=/arc/home/$CANFAR_USER/pp_switch
PP="$HOME/pilot-proxy"
FRZ="$HOME/pp_freeze"
BUNDLE="$FRZ/archive-local-65b49971ffa6"
VENV="$HOME/pp-venv-$(hostname)"

say "0. node identity"
hostname; nproc; free -g | head -2
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
python3 --version

say "0b. CUDA toolkit (image check)"
# The session image must be astroml-CUDA. astroml:latest carries the driver but
# no toolkit: nvidia-smi works, the kernel .so loads, fetching works, and the
# scan then dies on its first file inside cupy's JIT. Fail here instead.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "NO nvcc: this session has no CUDA toolkit."
  echo "  Its image is probably images.canfar.net/skaha/astroml:latest."
  echo "  Relaunch the session with images.canfar.net/skaha/astroml-cuda:latest"
  echo "  (the image notebooks 1 and 2 run) and re-run this script."
  exit 1
fi
echo "nvcc      : $(command -v nvcc)"

say "1. certificate"
CERT="$HOME/.ssl/cadcproxy.pem"
if [ ! -f "$CERT" ]; then
  echo "NO CERT at $CERT -- run:  cadc-get-cert -u $CANFAR_USER --days-valid 30"
  echo "then re-run this script."; exit 1
fi
chmod 600 "$CERT"
openssl x509 -in "$CERT" -noout -enddate

say "2. frozen source"
if [ ! -d "$PP/.git" ]; then
  git clone --quiet https://github.com/WVURAIL/pilotproxy "$PP"
fi
cd "$PP"
git fetch --quiet origin --tags
# $PP is on /arc and shared by every session. Moving HEAD while another
# session is scanning from it would change the code under a live run and
# fail every later gate (REV mismatch). So: no-op when already at REV;
# otherwise refuse unless explicitly allowed.
if [ "$(git rev-parse HEAD)" != "$REV" ]; then
  if [ "${PP_ALLOW_CHECKOUT:-}" = "YES" ]; then
    git checkout --quiet "$REV"
  else
    echo "REFUSING to move the shared checkout $PP from $(git rev-parse --short HEAD) to ${REV:0:7}:"
    echo "  other sessions may be scanning from it. If no scan is running anywhere,"
    echo "  re-run with PP_ALLOW_CHECKOUT=YES."
    exit 1
  fi
fi
test "$(git rev-parse HEAD)" = "$REV" || { echo "REV MISMATCH"; exit 1; }
test -z "$(git status --porcelain)" || { echo "TREE NOT CLEAN"; exit 1; }
echo "at $TAG = $REV (clean)"

rm -rf "$HOME/pp-venv"  # pp-venv-old-cleanup: v1 partial install
say "3. runtime restore (offline, from the freeze bundle wheelhouse)"
if [ ! -e "$BUNDLE/.restored_ok" ]; then
  echo "verifying bundle tar..."
  echo "$TARSHA  $SW/pp_runtime_freeze.tar" | sha256sum --check --strict || exit 1
  mkdir -p "$FRZ"
  tar -C "$FRZ" -xf "$SW/pp_runtime_freeze.tar"
  ( cd "$BUNDLE" && sha256sum --check --quiet SHA256SUMS ) || { echo "BUNDLE SHA256SUMS FAILED"; exit 1; }
  touch "$BUNDLE/.restored_ok"
fi
echo "bundle verified: $BUNDLE"
[ -e "$VENV/bin/activate" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
LOG=/tmp/pp_pip_install.log
pip install --no-index --find-links "$BUNDLE/wheelhouse" -r "$BUNDLE/requirements.lock" >"$LOG" 2>&1 \
  || { echo "OFFLINE LOCK INSTALL FAILED -- tail of $LOG:"; tail -25 "$LOG"; exit 1; }
grep -E '^Successfully installed' "$LOG" | tail -1
pip install --no-index --no-build-isolation -e . >>"$LOG" 2>&1 \
  || { echo "EDITABLE INSTALL FAILED -- tail of $LOG:"; tail -25 "$LOG"; exit 1; }
python -c "from pilot_proxy.provenance import package_source_sha256 as p; s=p(); assert s=='"$PKG"', s; print('package sha OK: '+s[:16]+'...')"

say "4. kernel build + digest for THIS node arch"
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
if [ "$SM" = "90" ] && [ -f "$QKLIB" ] && echo "$QKSHA  $QKLIB" | sha256sum --check --strict >/dev/null 2>&1; then
  echo "sm90 node: reusing the QUALIFIED kernel (no rebuild, shared cuda/ untouched)"
  KLIB="$QKLIB"; SHA="$QKSHA"
  echo "preserved: $KLIB"; echo "sha256   : $SHA"
  QUALIFIED=1
else
  echo "no qualified kernel for sm$SM -- building one. NOTE: a freshly built kernel is"
  echo "NOT qualified; its products will not interchange with the running shards until"
  echo "canfar_smoke_844.sh passes against it."
  make -C cuda clean >/dev/null; make -C cuda SM="$SM" >/dev/null || { echo "KERNEL BUILD FAILED"; exit 1; }
  QUALIFIED=0
fi
if [ "$QUALIFIED" = 0 ]; then
  SHA=$(sha256sum cuda/libpilotproxy.so | cut -d' ' -f1)
  KDIR="/arc/home/$CANFAR_USER/pp_kernels"
  mkdir -p "$KDIR"
  KLIB="$KDIR/pilotproxy-detector-core-2.3.0-sm${SM}-${SHA:0:12}.so"
  cp --no-clobber cuda/libpilotproxy.so "$KLIB"; chmod 555 "$KLIB" 2>/dev/null || true
  echo "preserved: $KLIB"
  echo "sha256   : $SHA"
fi

say "5. kernel gates (CPU reference + load check)"
if [ "$QUALIFIED" = 1 ]; then
  echo "(CPU reference test skipped: qualified kernel reused; it passed at qualification)"
else
  make -C cuda test_ref 2>&1 | tail -2
fi
PYTHONPATH=src python - "$KLIB" <<'PY'
import sys
from pilot_proxy.kernel import PilotProxyKernel
k = PilotProxyKernel(sys.argv[1])
print("kernel", k._get_version().as_string(), "fine:", k.supports_fine_powers(), "fused:", k.supports_fused_fine())
PY

if [ "${PP_SKIP_PROBE:-}" = "1" ]; then
  say "6. probe skipped (PP_SKIP_PROBE=1)"
  say "DONE"
  exit 0
fi
say "6. FETCH THROUGHPUT PROBE (the decision number)"
mkdir -p /tmp/pp_probe && cd /tmp/pp_probe
run_probe () {
  local workers=$1; shift
  local t0 t1 bytes=0
  t0=$(date +%s.%N)
  xargs -P "$workers" -I{} -a "$SW/probe_uris.txt" \
    sh -c 'cadcget --cert "$HOME/.ssl/cadcproxy.pem" "{}" -o "/tmp/pp_probe/$(basename {})" 2>/dev/null'
  t1=$(date +%s.%N)
  bytes=$(du -cb /tmp/pp_probe/*.h5 2>/dev/null | tail -1 | cut -f1)
  python3 -c "print(f'  workers=$workers  bytes={$bytes:,}  secs={$t1-$t0:.1f}  MiB/s={($bytes/1048576)/($t1-$t0):.1f}')"
  rm -f /tmp/pp_probe/*.h5
}
echo "single-stream:"; run_probe 1
echo "four-stream  :"; run_probe 4
echo "eight-stream :"; run_probe 8

say "DONE -- paste everything from '0. node identity' down"
