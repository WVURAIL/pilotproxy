#!/usr/bin/env bash
# CANFAR shard controller for the pilot-proxy v5 archive run at the frozen tag.
#   bash /arc/home/$CANFAR_USER/pp_switch/canfar_shard.sh <1|2|3> <update|gate|launch|resume|run|tripwire|status|stop>
#
# Shard 1 (notebook1): 506,521,537,552,568,583,675,752,767,813,829  (13.50 TiB, 86541 files)
# Shard 2 (notebook2): 598,614,629,644,660,690,706,721,736,783,798,844  (13.46 TiB, 85896 files)
# Shard 3 (notebook3): 767,783,813,829,798,844, the tail channels of shards 1 and 2,
#   merged back with canfar_merge_channel.sh as each completes (7.13 TiB, 48779 files)
# One shard runs per session and node. gate() checks the same frozen identity
# for every shard: source REV, package PKG, inventory e97a57f9..., weights WSHA
# and the qualified sm90 kernel 33b6e1c45c47 (cross-arch smoke PASS 2026-09-01).
#
# Predeclared dispositions (same as local, split by shard):
#   * sub-frame quarantines are expected: shard1 cap 2374, shard2 cap 2333
#     (total 4707), and shard3 cap 1406 on the channels it shares with them.
#   * --allow-partial stays off. Once only quarantined rows remain, the final
#     invocation of each shard exits nonzero with "incomplete requested scope",
#     and that exit is the designed endpoint of the shard.
#   * a cadcget "Not found" while cadcinfo answers is treated as an outage,
#     and the file is not counted as missing.
# launch requires PP_LAUNCH_CONFIRM=YES, resume requires PP_RESUME_CONFIRM=YES
# and run requires PP_RUN_CONFIRM=YES.
set -uo pipefail
# Site configuration, exported before running (see README.md):
#   CANFAR_USER  CANFAR account name. The kit and the runs are under /arc/home/$CANFAR_USER/.
: "${CANFAR_USER:?}"
umask 077
die(){ echo "SHARD-BLOCK: $*" >&2; exit 1; }
say(){ printf '\n===== %s =====\n' "$*"; }

SHARD="${1:-}"; MODE="${2:-status}"
case "$SHARD" in 1|2|3) ;; *) die "usage: canfar_shard.sh <1|2|3> <update|gate|launch|resume|run|tripwire|status|stop>";; esac

TAG=archive-run-source-20260829       # superseded by REV below
# Storage Inventory service to fetch through. Empty selects the library
# default (global raven), which is used whenever raven is healthy. Raven
# load-balances across replicas and measured 15-160 MiB/s, while a single
# pinned replica measured about 1.7 MiB/s sustained and timed out under load
# (2026-09-01). A replica is pinned only to ride out a raven outage:
#   PP_STORAGE_SERVICE=ivo://cadc.nrc.ca/uvic/minoc bash canfar_shard.sh ...
# raven status:  curl -s https://cadc-west-01.canfar.net/raven/availability
SVC="${PP_STORAGE_SERVICE-}"
export PILOT_PROXY_STORAGE_SERVICE="$SVC"
REV=b59b5c05fed2a9509a31e206f0911e76ca2d2885
PKG=3722012957975f7d5698c24ab3bf36b59ff26dd94fd84ae75b2eb0820d8ea34a
KLIB=/arc/home/$CANFAR_USER/pp_kernels/pilotproxy-detector-core-2.3.0-sm90-33b6e1c45c47.so
KSHA=33b6e1c45c472c65cf46031d4b009d6f6f96652b57c9bb362489f403dfeaedbd
INVSHA=e97a57f9349bcb44463d6fba9fcbfd71b03863fa5a44deca352910b59766be65
WSHA=1383c6d0ca521a26b317d008feb6e09eb41427155bda9a320f70bca62e0e6259
SW=/arc/home/$CANFAR_USER/pp_switch
PP="$HOME/pilot-proxy"
VENV="$HOME/pp-venv-$(hostname)"
INV="$SW/inventory.jsonl"
CERT="$HOME/.ssl/cadcproxy.pem"
MINOC_CAPS_1=https://ws-uv.canfar.net/minoc/capabilities
MINOC_CAPS_2=https://ws-cadc.canfar.net/minoc/capabilities

case "$SHARD" in
1)
  SEL=506,521,537,552,568,583,675,752,767,813,829
  SUBFRAME_CAP=2374
  GATE_URI=cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_100050005/baseband_100050005_506.h5
  GATE_MD5=62441de83c1b4f0f9b734f4264697425
  ;;
2)
  SEL=598,614,629,644,660,690,706,721,736,783,798,844
  SUBFRAME_CAP=2333
  GATE_URI=cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_100050005/baseband_100050005_844.h5
  GATE_MD5=9c082ed5909039f7809ac30c088a226a
  ;;
3)
  # A third session probed 150 MiB/s while shards 1 and 2 ran (2026-09-03),
  # so bandwidth is per session and a third shard adds throughput. Shard 3
  # takes the tail of each owner's list (767,813,829 from shard 1 and
  # 783,798,844 from shard 2), ordered by how soon the owner would otherwise
  # reach them. Finished channels are handed to the owner with
  # canfar_merge_channel.sh, so the owner skips them.
  # 48779 files, 7297.5 GiB and 1406 predeclared sub-frame quarantines.
  SEL=767,783,813,829,798,844
  SUBFRAME_CAP=1406
  GATE_URI=cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_100050005/baseband_100050005_844.h5
  GATE_MD5=9c082ed5909039f7809ac30c088a226a
  ;;
esac
GATE_BYTES=91311880
# Products carry the source identity that built them, and the detector
# refuses to append frames from a different build (detector_version embeds
# the package source hash). Each source therefore gets its own run directory.
# The products built from the 20260829 source (package e30ec73f...) stay
# untouched as evidence, and this build starts a clean run so that two
# implementations are never mixed in one product.
SRC_SHORT=${REV:0:7}
OUT=/arc/home/$CANFAR_USER/pp_runs/chime_pilots_rebuild_20260829_canfar_shard${SHARD}_${SRC_SHORT}
STG=/tmp/pp_stage_shard${SHARD}_${SRC_SHORT}
LOGDIR=/arc/home/$CANFAR_USER/pp_runs/logs
LOG=$LOGDIR/canfar_shard${SHARD}_${SRC_SHORT}.log

scan_args(){
  echo chime-scan \
    --source cadc-datatrail \
    --inventory "$INV" \
    --output-dir "$OUT" \
    --staging-dir "$STG" \
    --instrument chime \
    --analyzer pilot-proxy-detector \
    --select "$SEL" \
    --download-workers 8 --max-staged-files 16 --checkpoint-every 250 \
    --weights-path "$PP/weights/chime_dtv_weights_k128.bin" \
    --weight-coordinate-system post_spectral_sense_normalization \
    --lib-path "$KLIB" \
    --set fine_products=on
}

gate(){
  say "GATE shard $SHARD on $(hostname)"
  for OTHER in 1 2 3; do
    [ "$OTHER" = "$SHARD" ] && continue
    if pgrep -f "pilot-proxy chime-scan.*canfar_shard${OTHER}_" >/dev/null 2>&1; then
      test "${PP_ALLOW_COLOCATED:-}" = "YES" \
        || die "shard $OTHER is already running on THIS node ($(hostname)) -- one shard per node: run shard $SHARD in its own session (PP_ALLOW_COLOCATED=YES overrides)"
    fi
  done
  test -e "$VENV/bin/activate" || die "venv missing: $VENV -- run canfar_probe_bootstrap.sh on THIS node first"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  cd "$PP" || die "no checkout at $PP"
  test "$(git rev-parse HEAD)" = "$REV" || die "REV mismatch"
  test -z "$(git status --porcelain)" || die "tree not clean"
  got=$(python -c "from pilot_proxy.provenance import package_source_sha256 as p; print(p())")
  test "$got" = "$PKG" || die "package sha mismatch: $got"
  echo "$KSHA  $KLIB" | sha256sum --check --strict >/dev/null || die "kernel digest mismatch"
  echo "$INVSHA  $INV" | sha256sum --check --strict >/dev/null || die "inventory sha mismatch"
  echo "$WSHA  $PP/weights/chime_dtv_weights_k128.bin" | sha256sum --check --strict >/dev/null || die "weights sha mismatch"
  echo "identity  : source+package+kernel+inventory+weights all match frozen"

  test -f "$CERT" || die "no cert at $CERT -- run: cadc-get-cert -u $CANFAR_USER --days-valid 30"
  # CANFAR rewrites this shared file with a fresh 7-day delegated cert on every
  # session launch, so a 14-day floor would fail the gate after any new
  # session (it blocked shard 3 entirely). The supervisor resumes often, so
  # the cert only has to outlive the next stretch of work. The gate refuses
  # under 3 days and warns under 10.
  openssl x509 -in "$CERT" -noout -checkend $((3*86400)) >/dev/null \
    || die "cert expires within 3 days ($(openssl x509 -in "$CERT" -noout -enddate)) -- run: cadc-get-cert -u $CANFAR_USER --days-valid 30"
  if openssl x509 -in "$CERT" -noout -checkend $((10*86400)) >/dev/null; then
    echo "cert      : $(openssl x509 -in "$CERT" -noout -enddate)"
  else
    echo "cert      : $(openssl x509 -in "$CERT" -noout -enddate)  << under 10 days; re-mint when convenient (cadc-get-cert -u $CANFAR_USER --days-valid 30)"
  fi

  for url in "$MINOC_CAPS_1" "$MINOC_CAPS_2"; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$url") || die "capabilities probe failed: $url"
    test "$code" = "200" || die "minoc capabilities $url returned $code (outage not cleared)"
  done
  echo "minoc caps: both endpoints 200"

  rm -f /tmp/pp_gate_obj.h5
  if [ -n "$SVC" ]; then
    cadcget --cert "$CERT" -s "$SVC" "$GATE_URI" -o /tmp/pp_gate_obj.h5 \
      || die "byte gate failed via $SVC (replica down too? try PP_STORAGE_SERVICE=)"
  else
    cadcget --cert "$CERT" "$GATE_URI" -o /tmp/pp_gate_obj.h5 \
      || die "byte gate cadcget failed (outage? check raven/availability)"
  fi
  sz=$(stat -c %s /tmp/pp_gate_obj.h5)
  test "$sz" = "$GATE_BYTES" || die "byte gate size $sz != $GATE_BYTES"
  got=$(md5sum /tmp/pp_gate_obj.h5 | cut -d' ' -f1)
  test "$got" = "$GATE_MD5" || die "byte gate md5 $got != $GATE_MD5"
  rm -f /tmp/pp_gate_obj.h5
  echo "byte gate : $GATE_BYTES bytes, md5 verified (route: ${SVC:-default raven})"

  # cupy JIT-compiles at runtime and needs the CUDA toolkit headers. Without
  # them the scan dies on its first file (shard 3, 2026-09-03), so the gate
  # compiles one kernel before any scan starts.
  if [ -z "${CUDA_PATH:-}" ]; then
    for c in "${CUDA_HOME:-}" /usr/local/cuda /opt/conda              "$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /nonexistent/x)")")"; do
      [ -n "$c" ] && [ -r "$c/include/cuda.h" ] && { export CUDA_PATH="$c"; break; }
    done
  fi
  python - <<'PYCUDA' || die "cupy cannot compile a kernel (CUDA headers not found). Set CUDA_PATH to a toolkit whose include/cuda.h exists, or use a session image that ships the headers."
import os, sys
import cupy as cp
try:
    a = cp.arange(16, dtype=cp.float32)
    b = (a * 2).copy()            # forces an elementwise JIT, the op shard 3 died on
    cp.cuda.runtime.deviceSynchronize()
    assert float(b[1]) == 2.0
except Exception as exc:
    print(f"  cupy JIT FAILED (CUDA_PATH={os.environ.get('CUDA_PATH', '<unset>')}): "
          f"{type(exc).__name__}: {exc}", flush=True)
    sys.exit(1)
print(f"cupy JIT  : OK (CUDA_PATH={os.environ.get('CUDA_PATH', '<unset, found by cupy>')})")
PYCUDA

  free_kb=$(df -P /tmp | awk 'NR==2{print $4}')
  test "$free_kb" -ge $((50*1024*1024)) || die "/tmp has <50 GiB free ($((free_kb/1024/1024)) GiB)"
  echo "staging   : /tmp $((free_kb/1024/1024)) GiB free (need >=50)"
  echo "GATE: ALL PASS (shard $SHARD, select=$SEL)"
}

launch(){
  test "${PP_LAUNCH_CONFIRM:-}" = "YES" || die "set PP_LAUNCH_CONFIRM=YES to launch"
  gate
  say "LAUNCH shard $SHARD"
  test ! -e "$OUT" || die "output dir exists: $OUT (use resume mode)"
  test ! -e "$STG" || die "staging dir exists: $STG"
  mkdir -p "$STG" "$LOGDIR"
  echo "$(hostname) $(date -u +%FT%TZ) launch" >> "$OUT.hostlog" 2>/dev/null || true
  # setsid -f (foreground fork) keeps SIGINT at its default disposition in the
  # child. A bash async job (&) would start the scan with SIGINT ignored, and
  # every later kill -INT would do nothing (2026-09-01).
  # shellcheck disable=SC2046
  setsid -f pilot-proxy $(scan_args) >> "$LOG" 2>&1 < /dev/null
  sleep 1
  pid=$(pgrep -f "pilot-proxy chime-scan.*canfar_shard${SHARD}_" | head -1 || true)
  echo "launched pid ${pid:-unknown} on $(hostname); log: $LOG"
  echo "tripwire after the first checkpoint:  bash $SW/canfar_shard.sh $SHARD tripwire"
}

resume(){
  test "${PP_RESUME_CONFIRM:-}" = "YES" || die "set PP_RESUME_CONFIRM=YES to resume"
  gate
  say "RESUME shard $SHARD"
  test -d "$OUT" || die "nothing to resume: $OUT missing (use launch mode)"
  pgrep -f "pilot-proxy chime-scan.*$OUT" >/dev/null && die "scan already running for $OUT"
  mkdir -p "$STG" "$LOGDIR"
  echo "$(hostname) $(date -u +%FT%TZ) resume" >> "$OUT.hostlog" 2>/dev/null || true
  # setsid -f keeps SIGINT usable (see the comment in launch())
  # shellcheck disable=SC2046
  setsid -f pilot-proxy $(scan_args) >> "$LOG" 2>&1 < /dev/null
  sleep 1
  pid=$(pgrep -f "pilot-proxy chime-scan.*canfar_shard${SHARD}_" | head -1 || true)
  echo "resumed pid ${pid:-unknown} on $(hostname); log: $LOG"
}

tripwire(){
  say "TRIPWIRE shard $SHARD"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  npz=$(ls -t "$OUT/_per_pilot/"*.npz 2>/dev/null | head -1)
  test -n "$npz" || die "no product npz yet -- first checkpoint is 250 units, check again later"
  python - "$npz" "$PKG" "$KSHA" "$OUT/_per_pilot/quarantine.jsonl" "$SUBFRAME_CAP" <<'PYTRIP'
import json, os, sys
import numpy as np
npz, pkg, ksha, qpath, cap = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
with np.load(npz, allow_pickle=False) as z:
    dv = str(z["detector_version"])
    assert pkg in dv, "frozen package sha NOT stamped: " + dv
    assert ksha in dv, "qualified sm90 kernel sha NOT stamped: " + dv
    assert str(z["schema_version"]) == "pilotproxy_per_pilot_product_v6"
    order = [str(k) for k in z["unit_order"]]
    assert len(order) == len(set(order)), "duplicate units"
    pairs = list(zip(z["frame_unit_index"].tolist(), z["frame_in_unit"].tolist()))
    assert len(pairs) == len(set(pairs)), "duplicate frames"
    fine = z["fine_power_u64"]
    assert fine.dtype == np.uint64 and fine.shape[1:] == (3, 256), fine.shape
    # each frame power is the exact all-bin sum of its fine powers (schema v6)
    sums = fine.sum(axis=2, dtype=np.uint64)
    for term, key in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
        assert np.array_equal(z[key][:, 0], sums[:, term]), key + " is not the fine-power sum"
    assert int(z["rational_overflow_count"]) == 0
    valid = z["valid"]
    print("product          :", os.path.basename(npz))
    print("units / frames   :", len(order), "/", len(pairs))
    print("valid frames     : %d/%d (%.1f%%)" % (int(valid.sum()), len(valid), 100.0*valid.sum()/max(len(valid),1)))
    print("identity stamped : frozen source + qualified sm90 kernel")
# Quarantine classes:
#   sub-frame          predeclared, capped and expected
#   archive-truncated  stored object shorter than its own HDF5 superblock
#                      claims, so no reader can ever read it. The fault is in
#                      the archive copy and is permanent. Counted and capped,
#                      and the run continues.
#   other              fails the tripwire. This check caught the staging
#                      removal that quarantined readable units.
ARCHIVE_TRUNCATED_CAP = 50
n = sub = trunc = bad = 0
truncated_names = []
if os.path.exists(qpath):
    for line in open(qpath):
        if not line.strip():
            continue
        n += 1
        r = json.loads(line)
        reason = str(r.get("reason", "")) + str(r.get("detail", ""))
        name = r.get("unit_name", r.get("name", "?"))
        if "shorter than one transform" in reason:
            sub += 1
        elif "truncated file" in reason and "stored_eof" in reason:
            trunc += 1
            truncated_names.append(name)
        else:
            bad += 1
            if bad <= 3:
                print("UNEXPECTED quarantine:", name, reason[:90])
print("quarantine       : %d rows (cap %d) = %d sub-frame + %d archive-truncated"
      % (n, cap, sub, trunc))
if trunc:
    print("archive-truncated: %s%s"
          % (", ".join(truncated_names[:3]), " ..." if trunc > 3 else ""))
    print("                   (stored object shorter than its own HDF5 superblock;"
          " permanently unreadable, correctly quarantined)")
print("unexpected       : %d" % bad)
assert n <= cap, "quarantine over predeclared cap"
assert trunc <= ARCHIVE_TRUNCATED_CAP, (
    "archive-truncated objects over cap %d -- investigate, this should be rare"
    % ARCHIVE_TRUNCATED_CAP)
assert bad == 0, "unexpected quarantine class"
print("TRIPWIRE: PASS")
PYTRIP
}

run_foreground(){
  test "${PP_RUN_CONFIRM:-}" = "YES" || die "set PP_RUN_CONFIRM=YES to run"
  gate
  say "RUN (foreground) shard $SHARD"
  pgrep -f "pilot-proxy chime-scan.*$OUT" >/dev/null && die "scan already running for $OUT on this node"
  mkdir -p "$STG" "$LOGDIR"
  echo "$(hostname) $(date -u +%FT%TZ) run-foreground" >> "$OUT.hostlog" 2>/dev/null || true
  # In the foreground, this process is the session command, and a headless
  # session ends when it exits. The log still goes to /arc so every shard is
  # monitored the same way.
  exec >> "$LOG" 2>&1
  # shellcheck disable=SC2046
  exec pilot-proxy $(scan_args)
}

update(){
  say "UPDATE shard $SHARD source on $(hostname)"
  cd "$PP" || die "no checkout at $PP"
  pgrep -f "pilot-proxy chime-scan.*canfar_shard" >/dev/null 2>&1 \
    && die "a scan is running on this node -- stop it before changing source"
  git fetch --quiet origin
  git checkout --quiet "$REV" || die "checkout $REV failed"
  test "$(git rev-parse HEAD)" = "$REV" || die "REV mismatch after checkout"
  test -z "$(git status --porcelain)" || die "tree not clean"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  got=$(python -c "from pilot_proxy.provenance import package_source_sha256 as p; print(p())")
  test "$got" = "$PKG" || die "package sha mismatch after checkout: $got"
  echo "now at $REV, package sha $PKG"
  echo "storage service: ${SVC:-default raven}"
}

stop(){
  pid=$(pgrep -f "pilot-proxy chime-scan.*$OUT" | head -1 || true)
  test -n "${pid:-}" || { echo "no running scan for shard $SHARD on this node"; exit 0; }
  kill -INT "$pid"
  echo "SIGINT sent to pid $pid; waiting for exit..."
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "still running after 120 s (a pre-fix process ignores SIGINT); escalating to SIGTERM"
    kill -TERM "$pid"
    for i in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  fi
  kill -0 "$pid" 2>/dev/null && die "still running -- last resort is safe (checkpoint design is crash-proven):  kill -9 $pid"
  echo "stopped. staging left in place; safe to remove ONLY now:"
  echo "  rm -rf $STG"
}

status(){
  say "STATUS shard $SHARD @ $(date -u +%FT%TZ) on $(hostname)"
  pid=$(pgrep -f "pilot-proxy chime-scan.*$OUT" | head -1 || true)
  if [ -n "${pid:-}" ]; then
    echo "scan     : RUNNING pid $pid  elapsed $(ps -o etime= -p "$pid" | tr -d ' ')"
  else
    echo "scan     : NOT RUNNING"
  fi
  if [ -d "$OUT/_per_pilot" ]; then
    n=$(ls "$OUT/_per_pilot/"*.npz 2>/dev/null | wc -l)
    newest=$(ls -t "$OUT/_per_pilot/"*.npz 2>/dev/null | head -1)
    echo "products : $n channel product(s); newest: ${newest:-none} ($(date -u -r "$newest" +%FT%TZ 2>/dev/null || echo n/a))"
    q="$OUT/_per_pilot/quarantine.jsonl"
    [ -f "$q" ] && echo "quarant. : $(wc -l < "$q") rows (cap $SUBFRAME_CAP)"
  else
    echo "products : none yet"
  fi
  [ -d "$STG" ] && echo "staging  : $(find "$STG" -type f 2>/dev/null | wc -l) file(s) (bound 16)"
  df -P /tmp | awk 'NR==2{printf "tmp free : %.0f GiB\n", $4/1024/1024}'
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu      : /'
  openssl x509 -in "$CERT" -noout -checkend $((7*86400)) >/dev/null 2>&1 \
    && echo "cert     : >7 days remaining" || echo "cert     : EXPIRES WITHIN 7 DAYS -- renew before it lapses"
  [ -f "$LOG" ] && tail -2 "$LOG" | sed 's/^/log tail : /'
}

case "$MODE" in
  gate)     gate ;;
  launch)   launch ;;
  resume)   resume ;;
  tripwire) tripwire ;;
  run)      run_foreground ;;
  update)   update ;;
  stop)     stop ;;
  status)   status ;;
  *) die "unknown mode $MODE" ;;
esac
