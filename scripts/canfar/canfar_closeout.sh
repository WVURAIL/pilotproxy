#!/usr/bin/env bash
# Closeout for the 2026-09 CHIME archive run. The canonical per-pilot
# products are assembled into one run directory.
#
#   bash /arc/home/$CANFAR_USER/pp_switch/canfar_closeout.sh assemble
#
# The 23 canonical products of this run are split across two shard
# directories that do not overlap:
#   shard1 (11): 506 521 537 552 568 583 675 752 767 813 829
#   shard2 (12): 598 614 629 644 660 690 706 721 736 783 798 844
# The shard3 directory duplicates six of these channels. It is not copied
# and stays in place as reproduction evidence.
set -uo pipefail
# Site configuration, exported before running (see README.md):
#   CANFAR_USER  CANFAR account name. The kit and the runs are under /arc/home/$CANFAR_USER/.
: "${CANFAR_USER:?}"
say(){ printf '\n===== %s =====\n' "$*"; }
die(){ echo "CLOSEOUT-BLOCK: $*" >&2; exit 1; }

MODE="${1:-assemble}"
REV=b59b5c05fed2a9509a31e206f0911e76ca2d2885
PKG=3722012957975f7d5698c24ab3bf36b59ff26dd94fd84ae75b2eb0820d8ea34a
R=/arc/home/$CANFAR_USER/pp_runs
S1=$R/chime_pilots_rebuild_20260829_canfar_shard1_b59b5c0
S2=$R/chime_pilots_rebuild_20260829_canfar_shard2_b59b5c0
ALL=$R/chime_pilots_rebuild_20260829_ALL23
PP="$HOME/pilot-proxy"
VENV="$HOME/pp-venv-$(hostname)"
CH1="506 521 537 552 568 583 675 752 767 813 829"
CH2="598 614 629 644 660 690 706 721 736 783 798 844"

say "0. environment"
cd "$PP" || die "no checkout at $PP"
test "$(git rev-parse HEAD)" = "$REV" || die "REV mismatch"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
got=$(python -c "from pilot_proxy.provenance import package_source_sha256 as p; print(p())")
test "$got" = "$PKG" || die "package sha mismatch: $got"
echo "source  : $REV"
echo "package : $PKG"

say "1. assemble the 23 canonical products into one run directory"
test ! -e "$ALL" || die "$ALL exists; move it aside for a fresh assembly"
mkdir -p "$ALL/_per_pilot"
umask 077
n=0
for pair in "$S1:$CH1" "$S2:$CH2"; do
  src="${pair%%:*}"; chans="${pair#*:}"
  for c in $chans; do
    f="$src/_per_pilot/$c.npz"
    test -f "$f" || die "missing $f"
    cp "$f" "$ALL/_per_pilot/$c.npz"
    a=$(sha256sum "$f" | cut -d' ' -f1); b=$(sha256sum "$ALL/_per_pilot/$c.npz" | cut -d' ' -f1)
    test "$a" = "$b" || die "copy of $c.npz did not verify"
    n=$((n+1))
  done
done
test "$n" -eq 23 || die "expected 23, copied $n"
echo "products: $n copied and sha256-verified"

# One quarantine ledger for the assembled run, deduplicated by key.
python - "$S1/_per_pilot/quarantine.jsonl" "$S2/_per_pilot/quarantine.jsonl" \
         "$ALL/_per_pilot/quarantine.jsonl" <<'PYQ'
import json, os, sys
src1, src2, dst = sys.argv[1], sys.argv[2], sys.argv[3]
seen, rows = set(), []
for p in (src1, src2):
    if not os.path.exists(p):
        continue
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get("quarantine_key")
        if k in seen:
            continue
        seen.add(k); rows.append(r)
with open(dst, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
sub = sum(1 for r in rows if "shorter than one transform" in r["reason"])
other = len(rows) - sub
print(f"quarantine: {len(rows)} rows ({sub} sub-frame + {other} archive-corrupt), 0 duplicates")
PYQ

say "2. verify the assembled set"
python - "$ALL/_per_pilot" <<'PYV'
import sys, pathlib
import numpy as np
d = pathlib.Path(sys.argv[1])
files = sorted(d.glob("*.npz"), key=lambda p: int(p.stem))
vers, chans = set(), []
for f in files:
    with np.load(f, allow_pickle=False) as z:
        vers.add(str(z["detector_version"]))
        fid = int(np.asarray(z["freq_id"]).reshape(-1)[0])
        assert fid == int(f.stem), f"{f.name}: holds freq_id {fid}"
        chans.append(fid)
assert len(vers) == 1, f"{len(vers)} distinct detector_version -- not one build"
print(f"  {len(files)} products, channels {chans[0]}..{chans[-1]}, one detector_version")
print(f"  {next(iter(vers))[:90]}...")
PYV

say "DONE -- assembled at:"
echo "  $ALL"
echo
echo "Its _per_pilot/ products feed the threshold pipeline (docs/CHIME_RUN_WORKFLOW.md)."
