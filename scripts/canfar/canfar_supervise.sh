#!/usr/bin/env bash
# Keep one shard running across archive outages and transient fetch failures.
#
#   setsid -f bash /arc/home/$CANFAR_USER/pp_switch/canfar_supervise.sh 1
#   (shard 1 on notebook1, shard 2 on notebook2, shard 3 on notebook3)
#
# A download worker that dies takes the whole scan with it, and not every
# transient is retried in fetch(). One such error is a raw urllib3
# ProtocolError ("Connection broken: IncompleteRead"), which escapes
# expected_errors(). The archive also has outages measured in hours (two in
# the 24 h before this script was written). A change to the frozen source
# mid-run would orphan every product built so far, so the source is left
# unchanged and the scan is restarted by this script. The scan is resumable
# by design, and a restart costs only the units in flight.
#
# Two cases are handled differently:
#   * gate refused (archive down, cert, colocation). No scan ran, so the
#     supervisor waits with a growing backoff and tries again. An outage is
#     ridden out this way, and it is never counted as a failed run.
#   * scan ran and exited. The scan is resumed as soon as it is gone. If it
#     made no progress, the supervisor backs off before the next attempt so a
#     flapping archive is not hammered, and any progress resets the backoff.
# The supervisor exits 0 only at the predeclared terminal state of the run
# ("incomplete requested scope", with only quarantined rows left). Stop it with:
#   pkill -f canfar_supervise
set -uo pipefail
# Site configuration, exported before running (see README.md):
#   CANFAR_USER  CANFAR account name. The kit and the runs are under /arc/home/$CANFAR_USER/.
: "${CANFAR_USER:?}"

SHARD="${1:-}"
case "$SHARD" in 1|2|3) ;; *) echo "usage: canfar_supervise.sh <1|2|3>"; exit 2;; esac

SW=/arc/home/$CANFAR_USER/pp_switch
REV=b59b5c05fed2a9509a31e206f0911e76ca2d2885
SRC_SHORT=${REV:0:7}
OUT=/arc/home/$CANFAR_USER/pp_runs/chime_pilots_rebuild_20260829_canfar_shard${SHARD}_${SRC_SHORT}
LOG=/arc/home/$CANFAR_USER/pp_runs/logs/canfar_shard${SHARD}_${SRC_SHORT}.log
SLOG=/arc/home/$CANFAR_USER/pp_runs/logs/supervise_shard${SHARD}.log
MAX_ATTEMPTS=400          # resume attempts of either kind, about a week at the backoff cap
BACKOFF_MIN=60
BACKOFF_MAX=1800

say(){ echo "$(date -u +%FT%TZ) [sup$SHARD] $*" >> "$SLOG"; }
scan_running(){ pgrep -f "pilot-proxy chime-scan.*canfar_shard${SHARD}_" >/dev/null 2>&1; }
# durable product bytes, used as the progress measure
progress(){ find "$OUT/_per_pilot" -name '*.npz' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
terminal(){ tail -40 "$LOG" 2>/dev/null | grep -q 'incomplete requested scope'; }

say "supervisor started on $(hostname) for shard $SHARD (durable bytes: $(progress), run dir $([ -d "$OUT" ] && echo present || echo absent))"

attempts=0
backoff=$BACKOFF_MIN
while [ "$attempts" -lt "$MAX_ATTEMPTS" ]; do

    if scan_running; then
        sleep 120
        continue
    fi

    if terminal; then
        say "predeclared terminal state reached (incomplete requested scope); supervisor exiting 0"
        exit 0
    fi

    attempts=$((attempts + 1))
    before=$(progress)
    # A shard that has never run has no run directory, and resume refuses to
    # run without one, so such a shard is launched. Launch runs the same gate
    # chain and the same scan, and its own fresh-path guard keeps it from
    # starting over an existing run.
    if [ -d "$OUT" ]; then
        MODE=resume; CONFIRM=PP_RESUME_CONFIRM
    else
        MODE=launch; CONFIRM=PP_LAUNCH_CONFIRM
    fi
    say "attempt #$attempts: $MODE (durable bytes $before, backoff ${backoff}s)"

    if ! env "$CONFIRM=YES" bash "$SW/canfar_shard.sh" "$SHARD" "$MODE" >> "$SLOG" 2>&1; then
        # gate refused and nothing ran, so wait it out without counting a failed run
        say "gate refused (archive outage / cert / colocation) -- waiting ${backoff}s"
        sleep "$backoff"
        backoff=$(( backoff * 2 > BACKOFF_MAX ? BACKOFF_MAX : backoff * 2 ))
        continue
    fi

    # gate accepted, so wait for the detached scan to appear and then to exit
    sleep 30
    while scan_running; do sleep 120; done
    after=$(progress)
    if [ "$after" -gt "$before" ]; then
        say "scan exited after progress ($before -> $after bytes); resuming promptly"
        backoff=$BACKOFF_MIN
    else
        say "scan exited with NO progress ($before bytes) -- likely transient at start; backing off ${backoff}s"
        sleep "$backoff"
        backoff=$(( backoff * 2 > BACKOFF_MAX ? BACKOFF_MAX : backoff * 2 ))
    fi
done

say "hit MAX_ATTEMPTS=$MAX_ATTEMPTS; stopping so a human looks"
exit 1
