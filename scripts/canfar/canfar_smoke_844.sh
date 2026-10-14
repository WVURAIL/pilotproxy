#!/usr/bin/env bash
# CANFAR cross-arch smoke -- pilot-proxy v5 at the frozen tag.
# Run in notebook1 AFTER the probe bootstrap succeeded:
#   bash /arc/home/$CANFAR_USER/pp_switch/canfar_smoke_844.sh
# Repeats the local B2d rehearsal scan shape (channel 844, first 8 files,
# production worker profile) on the H100/sm90 kernel, then compares the
# product against the local sm89 rehearsal product unit-by-unit.
# v5 coarse/fine terms are exact integers: cross-arch agreement must be
# bit-for-bit; only psd_frame_db_i16 is allowed +-1 code (0.01 dB).
set -uo pipefail
# Site configuration: the CANFAR account name; export it before running (see README.md).
#   CANFAR_USER  the CANFAR account; the kit and the runs live under /arc/home/$CANFAR_USER/
: "${CANFAR_USER:?}"
say(){ printf '\n===== %s =====\n' "$*"; }

REV=b59b5c05fed2a9509a31e206f0911e76ca2d2885
SW=/arc/home/$CANFAR_USER/pp_switch
PP="$HOME/pilot-proxy"
VENV="$HOME/pp-venv-$(hostname)"
KLIB=/arc/home/$CANFAR_USER/pp_kernels/pilotproxy-detector-core-2.3.0-sm90-33b6e1c45c47.so
KSHA=33b6e1c45c472c65cf46031d4b009d6f6f96652b57c9bb362489f403dfeaedbd
REF="$SW/ref_844_local_sm89.npz"
REFSHA=13364564ef2ac396b4a3c0c1bcb4edbfbbf6d545f8b2ba0bdb34f8758776a7c6
OUTDIR=/arc/home/$CANFAR_USER/pp_runs/canfar_smoke_844_b59b5c0
SVC="${PP_STORAGE_SERVICE:-ivo://cadc.nrc.ca/uvic/minoc}"
export PILOT_PROXY_STORAGE_SERVICE="$SVC"
STG=/tmp/pp_smoke_stage

say "0. environment (storage service: ${SVC:-default raven})"
cd "$PP"
test "$(git rev-parse HEAD)" = "$REV" || { echo "REV MISMATCH"; exit 1; }
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -c "from pilot_proxy.provenance import package_source_sha256 as p; print('package sha', p()[:16])"
echo "$KSHA  $KLIB" | sha256sum --check --strict || exit 1
echo "$REFSHA  $REF" | sha256sum --check --strict || exit 1

say "1. cuda tree identity vs preserved-kernel manifest"
sha256sum --check --strict <<'SUMS' || { echo "CUDA TREE DRIFT vs sm89 manifest"; exit 1; }
805b87cb7cedf518957bce127a17c3df4f0d22fbb008519979d7484bae00b13e  cuda/Makefile
c675a4c517f6abe9d3ca4121e3507c741fa7030da20404c51025028ace30365d  cuda/config.h
9cc1d39468e1eef3aff5e65384b9cb725abf659766214a6cfa73ede5e783389a  cuda/f_statistic.cu
61182c7610ac99856d6aed746d64893ea1f2eb1e171512ea1ce331785ae060a8  cuda/f_statistic.h
006516487af814f0dfd4458b87b341fbf90e76bbf4986e1066aa207eeec45b69  cuda/fxfft256_ref.c
SUMS
echo "cuda tree identical to the tree that built the sm89 production kernel"

say "2. kernel gate (corrected: pilot_proxy.kernel)"
python - "$KLIB" <<'PYGATE'
import sys
from pilot_proxy.kernel import FStatKernel
k = FStatKernel(sys.argv[1])
v = k._get_version(); sp = k._get_specs(); ft = k._get_features()
print("version :", v.as_string())
print("specs   :", sp.as_descriptive_dict())
print("features:", ft.as_dict())
print("fine    :", k.get_fine_specs())
assert v.as_string() == "2.3.0", v.as_string()
assert (sp.K, sp.N, sp.bits, sp.reference_offset_bins) == (128, 3, 4, 2)
assert ft.use_dp4a and ft.use_uint64_power_accumulation
assert k.supports_fine_powers() and k.supports_fused_fine()
fine = k.get_fine_specs()
assert {128, 2, 256} <= set(int(x) for x in fine.values()), fine
print("KERNEL GATE: PASS")
PYGATE

say "3. smoke scan (channel 844, first 8 files, production worker profile)"
test ! -e "$OUTDIR" || { echo "SMOKE-BLOCK: $OUTDIR exists (fresh smoke only) -- move it aside first"; exit 1; }
mkdir -p "$STG"
umask 077
pilot-proxy chime-scan \
  --source cadc-datatrail \
  --inventory "$SW/inventory.jsonl" \
  --output-dir "$OUTDIR" \
  --staging-dir "$STG" \
  --instrument chime \
  --analyzer pilot-proxy-detector \
  --select 844 \
  --download-workers 4 --max-staged-files 8 --checkpoint-every 2 \
  --weights-path "$PP/weights/chime_dtv_weights_k128.bin" \
  --weight-coordinate-system post_spectral_sense_normalization \
  --lib-path "$KLIB" \
  --set fine_products=on \
  --max-files 8 --allow-partial || { echo "SCAN FAILED"; exit 1; }

say "4. validate"
pilot-proxy validate-products --run-dir "$OUTDIR" || { echo "VALIDATE FAILED"; exit 1; }

say "5. cross-arch comparison (sm89 local vs sm90 canfar, keyed by unit)"
python - "$REF" "$OUTDIR/_per_pilot/844.npz" <<'PYCMP'
import sys
import numpy as np

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

A = load(sys.argv[1])   # local sm89 reference
B = load(sys.argv[2])   # canfar sm90

ka = [str(k) for k in A["unit_order"]]; kb = [str(k) for k in B["unit_order"]]
shared = sorted(set(ka) & set(kb))
print("units : local=%d canfar=%d shared=%d" % (len(ka), len(kb), len(shared)))
assert shared, "no shared units -- inventories diverged too far for this smoke"

def fmap(d, ko):
    return {(ko[int(u)], int(f)): i
            for i, (u, f) in enumerate(zip(d["frame_unit_index"], d["frame_in_unit"]))}
fa, fb = fmap(A, ka), fmap(B, kb)
pairs = sorted(k for k in fa if k in fb and k[0] in shared)
print("frames: local=%d canfar=%d matched=%d" % (len(fa), len(fb), len(pairs)))
assert pairs, "no matched frames"
ia = [fa[k] for k in pairs]; ib = [fb[k] for k in pairs]
ua = {k: i for i, k in enumerate(ka)}; ub = {k: i for i, k in enumerate(kb)}
sua = [ua[k] for k in shared]; sub = [ub[k] for k in shared]
n_fa, n_fb = len(A["frame_unit_index"]), len(B["frame_unit_index"])
n_ua, n_ub = len(ka), len(kb)
if n_fa == n_ua or n_fb == n_ub:
    print("note: frame count == unit count on one side; axis chosen frame-first")

SKIP = {"unit_order", "unit_keys", "frame_unit_index", "frame_in_unit",
        "frame_index", "detector_version", "_datatrawl_manifest"}
TOL1 = {"psd_frame_db_i16"}
only = sorted(set(A) ^ set(B))
if only:
    print("keys not in both products:", only)

fails = []
for k in sorted(set(A) & set(B)):
    if k in SKIP:
        continue
    a, b = A[k], B[k]
    if a.ndim == 0 or a.shape[0] not in (n_fa, n_ua):
        same = a.shape == b.shape and bool(np.array_equal(a, b))
        print("  %-36s [meta ] %s" % (k, "EQ" if same else "DIFF"))
        if not same:
            fails.append((k, "metadata differs"))
        continue
    if a.shape[0] == n_fa and b.shape[0] == n_fb:
        x, y, axis = a[ia], b[ib], "frame"
    elif a.shape[0] == n_ua and b.shape[0] == n_ub:
        x, y, axis = a[sua], b[sub], "unit "
    else:
        fails.append((k, "axis mismatch %s vs %s" % (a.shape, b.shape)))
        continue
    if x.shape != y.shape:
        fails.append((k, "shape %s vs %s" % (x.shape, y.shape)))
        continue
    if k in TOL1:
        d = np.abs(x.astype(np.int32) - y.astype(np.int32))
        mx = int(d.max()) if d.size else 0
        n1 = int(np.count_nonzero(d)) if d.size else 0
        ok = mx <= 1
        print("  %-36s [%s] max|d|=%d code(s), %d elements differ %s"
              % (k, axis, mx, n1, "OK" if ok else "FAIL"))
        if not ok:
            fails.append((k, "max %d codes" % mx))
    elif np.issubdtype(x.dtype, np.integer) or x.dtype == bool or x.dtype.kind in "SU":
        same = bool(np.array_equal(x, y))
        print("  %-36s [%s] %s" % (k, axis, "EXACT" if same else "DIFF"))
        if not same:
            fails.append((k, "integer/string field differs"))
    else:
        d = np.abs(x - y)
        s = np.maximum(np.abs(x), np.abs(y)); s[s == 0] = 1
        r = float((d / s).max()) if d.size else 0.0
        ok = r <= 1e-6
        print("  %-36s [%s] float max rel %.3e %s" % (k, axis, r, "OK" if ok else "FAIL"))
        if not ok:
            fails.append((k, "rel %.3e" % r))

print("local  detector_version:", str(A.get("detector_version", "?"))[:100])
print("canfar detector_version:", str(B.get("detector_version", "?"))[:100])
if fails:
    print("CROSS-ARCH SMOKE: FAIL")
    for k, why in fails:
        print("  -", k, ":", why)
    sys.exit(1)
print("CROSS-ARCH SMOKE: PASS -- sm89 and sm90 products agree on all matched units/frames")
PYCMP

say "6. timing (for shard arithmetic)"
python - "$OUTDIR/stats.json" <<'PYSTATS'
import json, sys
s = json.load(open(sys.argv[1]))
print(json.dumps(s, indent=1)[:1200])
PYSTATS

say "DONE -- paste everything from '0. environment' down"
