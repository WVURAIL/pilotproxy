#!/bin/bash
# Switch the Pathfinder DTV mask on and off in blocks through choco, with recording on, for a run that
# compares the data with and without the mask. Run it on the choco host, inside tmux:
#
#     tmux new -s dtvmask 'bash pathfinder_mask_schedule.sh 2>&1 | tee -a mask_schedule.log'
#
# Recording goes on and the mask off at the start. At the end of each block the mask switches (off, on,
# off, ...), and after the last block the mask and recording go off. Each mask switch carries
# valid_at_time_ns LEAD seconds ahead, so every X-engine switches at the same frame. The visibilities
# record frac_dtv_mask_enabled, so each bin is labelled whatever the switch times.
#
# Settings, from the environment:
#   BLOCK   seconds per block (600)          BLOCKS  number of blocks (6; even gives as many on as off)
#   LEAD    seconds of lead per switch (20)   TARGET  choco group or node, e.g. cx/cx47 (cx)
#   CHOCO   the choco command (choco)
set -u
CHOCO=${CHOCO:-choco}
TARGET=${TARGET:-cx}
BLOCK=${BLOCK:-600}
BLOCKS=${BLOCKS:-6}
LEAD=${LEAD:-20}
stamp() { date -u +%FT%TZ; }

mask() {
    local at=$(( ($(date +%s) + LEAD) * 1000000000 ))
    echo "$(stamp) mask $1 from valid_at_time_ns $at"
    "$CHOCO" set "$TARGET" dtv_mask_enabled "{\"enabled\": $1, \"valid_at_time_ns\": $at}" \
        || echo "$(stamp) choco set dtv_mask_enabled $1 FAILED"
}

write() {
    echo "$(stamp) recording $1"
    "$CHOCO" set "$TARGET" dtv_write_enabled "{\"enabled\": $1}" \
        || echo "$(stamp) choco set dtv_write_enabled $1 FAILED"
}

wait_until() {
    local now
    now=$(date +%s)
    [ "$1" -gt "$now" ] && sleep $(($1 - now))
}

mask false
write true
start=$(date +%s)
for ((k = 1; k < BLOCKS; k++)); do
    wait_until $((start + k * BLOCK - LEAD))
    if ((k % 2)); then mask true; else mask false; fi
done
wait_until $((start + BLOCKS * BLOCK - LEAD))
mask false
sleep "$LEAD"
write false
echo "$(stamp) done"
