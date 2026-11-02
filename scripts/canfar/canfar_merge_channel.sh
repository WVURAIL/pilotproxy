#!/usr/bin/env bash
# Hand a channel finished by shard 3 to the shard that owns it in its
# --select list, so the owner skips the channel when it reaches it.
#
#   bash /arc/home/$CANFAR_USER/pp_switch/canfar_merge_channel.sh <channel>
#   (runs on any session, since it works on /arc and takes the same lock as the scans)
#
# The per-channel product carries the full detector provenance (source hash,
# kernel hash, weights, contract), and the owner's resume() refuses anything
# that does not match its own. All three shards run the same frozen source and
# the same qualified kernel, so a product built by shard 3 is indistinguishable
# from one the owner would have built. Once the product and the quarantine rows
# of that channel are in the owner's directory, the owner finds every unit
# already disposed and prints
# "nothing to do -- selection already complete in this product".
#
# The merge is refused unless shard 3 has fully disposed the channel (no failed
# and no unprocessed units), the owner has no product for it yet, the owner is
# not working on it, and the provenance matches an existing owner product.
set -euo pipefail
# Site configuration, exported before running (see README.md):
#   CANFAR_USER  CANFAR account name. The kit and the runs are under /arc/home/$CANFAR_USER/.
: "${CANFAR_USER:?}"
CH="${1:-}"
case "$CH" in 767|813|829) OWNER=1 ;; 783|798|844) OWNER=2 ;;
  *) echo "usage: canfar_merge_channel.sh <767|813|829|783|798|844>"; exit 2 ;; esac
die(){ echo "MERGE-BLOCK: $*" >&2; exit 1; }

SW=/arc/home/$CANFAR_USER/pp_switch          # holds inventory.jsonl
R=/arc/home/$CANFAR_USER/pp_runs
SRC=$R/chime_pilots_rebuild_20260829_canfar_shard3_b59b5c0
DST=$R/chime_pilots_rebuild_20260829_canfar_shard${OWNER}_b59b5c0
SRCP=$SRC/_per_pilot; DSTP=$DST/_per_pilot
LOGDIR=$R/logs
VENV="$HOME/pp-venv-$(hostname)"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "== merge channel $CH: shard3 -> shard$OWNER =="

# 1. shard 3 has fully disposed the channel
test -f "$SRCP/$CH.npz" || die "no product $SRCP/$CH.npz"
python - "$SRC/scan_scope.json" "$CH" <<'PY' || die "shard 3 has not fully disposed channel (see above)"
import json, os, sys
scope = json.load(open(sys.argv[1])); ch = int(sys.argv[2])


def entry_channel(p):
    """Scope entries name their channel by product path, not a freq_id key."""
    prod = p.get("product") or ""
    base = os.path.basename(prod)
    if base.endswith(".npz") and base[:-4].isdigit():
        return int(base[:-4])
    sel = p.get("selection")
    if isinstance(sel, (list, tuple)) and len(sel) == 1:
        return int(sel[0])
    if isinstance(sel, int):
        return int(sel)
    return None


ent = next((p for p in scope["pilots"] if entry_channel(p) == ch), None)
assert ent, "channel not in shard-3 scope"
# A channel with quarantines reports status "partial" by design (the
# predeclared disposition of the run). The channel is complete when every
# enumerated unit is completed or quarantined and no unit is failed or
# unprocessed.
assert int(ent.get("enumerated", -1)) == (
    int(ent.get("completed", 0)) + int(ent.get("quarantined", 0))
), "completed + quarantined does not account for every enumerated unit"
print(f"  shard3 scope: status={ent.get('status')} completed={ent.get('completed')} "
      f"quarantined={ent.get('quarantined')} failed={ent.get('failed')} unprocessed={ent.get('unprocessed')}")
assert int(ent.get("failed", 1)) == 0, "failed units remain (retryable) -- let shard 3 finish"
assert int(ent.get("unprocessed", 1)) == 0, "unprocessed units remain -- let shard 3 finish"
PY

# 2. owner has no product yet and is not on this channel
test ! -e "$DSTP/$CH.npz" || die "owner already has $DSTP/$CH.npz"
last=$(grep 'select=\[' "$LOGDIR/canfar_shard${OWNER}_b59b5c0.log" | tail -1 | grep -oE 'select=\[[0-9]+\]' || true)
[ "$last" = "select=[$CH]" ] && die "owner shard$OWNER is currently working channel $CH -- too late to merge"
if [ -e "$DSTP/.$CH.npz.datatrawl.lock" ]; then
  flock -n "$DSTP/.$CH.npz.datatrawl.lock" true || die "owner holds the lock on $CH -- it is working it"
fi
echo "  owner shard$OWNER: no product for $CH, currently on ${last:-?}"

# 3. run-wide provenance must match an owner product, and the product must
#    be this channel. weights_hash, target_norm_sq and reference_norm_sum_sq
#    are functions of the weights evaluated for one channel, so they differ
#    between channels by construction and are not compared. The weight bank
#    and manifest hashes are the same for the whole run, and they are checked.
ref=$(ls "$DSTP"/*.npz | head -1)
python - "$SRCP/$CH.npz" "$ref" "$CH" "$SW/inventory.jsonl" <<'PY' || die "provenance mismatch -- do not merge"
import json
import sys

import numpy as np

RUN_WIDE = ("detector_version", "weight_bank_sha256", "weight_manifest_sha256",
            "mask_rule", "schema_version", "schema_name", "schema_revision",
            "source_event_key_schema_version")


def fields(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: str(z[k]) for k in keys if k in z.files}


src, ref, ch, inv_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
a, b = fields(src, RUN_WIDE), fields(ref, RUN_WIDE)
shared = sorted(set(a) & set(b))
bad = [k for k in shared if a[k] != b[k]]
assert not bad, "run-wide identity differs in: " + ", ".join(bad)
assert "detector_version" in shared, "detector_version missing from one product"
print(f"  run-wide identity: matches owner's product on {len(shared)} field(s) "
      f"including detector_version")

# the product must be the channel we are merging, checked against the inventory
with np.load(src, allow_pickle=False) as z:
    got = int(np.asarray(z["freq_id"]).reshape(-1)[0])
    hz = float(np.asarray(z["chime_frequency_hz"]).reshape(-1)[0])
assert got == ch, f"product is channel {got}, not {ch}"

# Exact CHIME center frequency of the channel. Products carry it unrounded, so
# the product must match it to better than 1 Hz. The inventory freq_mhz is
# rounded to 4 decimals (100 Hz), so it is used only as a cross-check to that
# resolution.
exact = (800.0 - 400.0 * ch / 1024.0) * 1e6
assert abs(hz - exact) < 1.0, (
    f"centre {hz} Hz != exact {exact} Hz for channel {ch}")

want = None
for line in open(inv_path):
    if not line.strip():
        continue
    r = json.loads(line)
    if int(r["freq_id"]) == ch:
        want = float(r["freq_mhz"]) * 1e6
        break
assert want is not None, f"channel {ch} absent from the inventory"
assert abs(exact - want) <= 100.0, (
    f"inventory {want} Hz disagrees with the formula {exact} Hz beyond its "
    f"4-decimal rounding for channel {ch}")
print(f"  channel identity : freq_id {got}, centre {hz/1e6:.6f} MHz exact; "
      f"inventory agrees within its rounding")
PY

# 4. atomic copy of the product, verified
tmp="$DSTP/.$CH.npz.merge.tmp"
cp "$SRCP/$CH.npz" "$tmp"; chmod 600 "$tmp"
s1=$(sha256sum "$SRCP/$CH.npz" | cut -d' ' -f1); s2=$(sha256sum "$tmp" | cut -d' ' -f1)
[ "$s1" = "$s2" ] || { rm -f "$tmp"; die "copy verification failed"; }
mv "$tmp" "$DSTP/$CH.npz"
echo "  product: copied, sha256 $s1"

# 5. append this channel's quarantine rows under the scan's own lock, deduplicated
n=$(python - "$SRCP/quarantine.jsonl" "$DSTP/quarantine.jsonl" "$CH" "$DSTP/.quarantine.jsonl.datatrawl.lock" <<'PY'
import fcntl, json, os, sys
src, dst, ch, lock = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
suffix = f"_{ch}.h5"
rows = [json.loads(l) for l in open(src) if l.strip()] if os.path.exists(src) else []
rows = [r for r in rows if str(r.get("unit_name", "")).endswith(suffix)]
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
try:
    have = set()
    if os.path.exists(dst):
        for l in open(dst):
            if l.strip():
                have.add(json.loads(l).get("quarantine_key"))
    new = [r for r in rows if r.get("quarantine_key") not in have]
    with open(dst, "a") as fh:
        for r in new:
            fh.write(json.dumps(r) + "\n")
        fh.flush(); os.fsync(fh.fileno())
finally:
    fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
print(len(new))
PY
)
echo "  quarantine: appended $n row(s) for channel $CH to owner ledger (locked, deduplicated)"

# 6. provenance trail in the owner's directory
echo "{\"channel\": $CH, \"from\": \"shard3\", \"sha256\": \"$s1\", \"quarantine_rows\": $n, \"merged_at_utc\": \"$(date -u +%FT%TZ)\", \"by\": \"$(hostname)\"}" \
  >> "$DST/merged_from_shard3.jsonl"
echo "MERGE OK: channel $CH now owned by shard$OWNER; it will skip it on arrival"
