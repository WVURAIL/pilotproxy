# Design note: automatic recalibration

Status: future work, not built. Today calibration is the manual loop of
[CALIBRATION_RUNBOOK.md](CALIBRATION_RUNBOOK.md). This note records the design proposed in the dissertation
(Sec. 6.3; repository WVURAIL/dissertation, private to the group) so that whoever builds it starts from the
same plan.

## Why

- The kept shelf drifts within an era. In the archive it moves by about 0.2 to 2.3 shelf limits per year.
- Thresholds pooled over a whole era failed on its last 12 months; thresholds re-selected on those months
  passed. A bundle calibrated once ages.
- A live node decides about 24 frames per second per channel, about 2 million per day, far more than any
  recording that is calibrated by hand.

## Accumulator

Per channel and per candidate threshold, keep sums per sidereal-day bin:

- kept frame count `n`
- sum and sum of squares of the frame-level excess `e = F / mu0 - 1`, with `F = 2 P0 / (P1 + P2)` from the
  three frame powers the kernel already forms
- for the control (channel 37): its null and selection-shift statistics, the inputs `o` and `delta(u)` of the
  power-limit test

Sums add, so any window is formed afterwards from the day bins: the last day, the last 2 to 4 weeks, since the
last era change, or the whole history.

## Triggers

| Trigger | Watches | Action | Needs |
|---|---|---|---|
| 1. Null | Control masked share over the last sidereal day, against its usual spread (gains, inputs, RFI, Sun) | Recalibrate the default word | Today's published pair (`dtv_stat`) |
| 2. Kept shelf | Running kept excess plus its margin, approaching the power limit | Re-select the BAO threshold | Per-frame excess (the three frame powers) |
| 3. Transmitter | Jumps in a channel's masked share: on/off, or an ATSC 3.0 conversion, where the pilot vanishes and the masked share drops to near zero | Flag, check station records, apply the era rule | Today's published pair |

## Windows

- 1 sidereal day for the null check (trigger 1) and transmitter jumps (trigger 3).
- About 2 to 4 weeks of sidereal days for the kept-shelf trigger and re-selection (trigger 2).
- Choose the kept-shelf window from data. Once about a month exists, compute the kept-shelf margin for 1, 7,
  14 and 28-day windows and take the shortest whose margin sits well below the limit on every usable channel.
- Frames within a day are not independent. The independent unit is closer to a sidereal day, so errors are
  clustered by day, not counted by frame.

## Missing pieces

| Piece | Where | Rough size |
|---|---|---|
| Publish and record the three frame powers per frame | Kotekan | Small |
| Rolling per-day accumulator | PilotProxy | Medium |
| Port the measured-coupling BAO test (noise and remainder tests, coupling records, RFIsher-dependent budgets) from the dissertation scripts | PilotProxy | Largest software piece |
| Threshold words updatable at run time and synchronized across nodes, like the mask switch of kotekan#1927 | Kotekan | Medium |
| Policy: triggers, rate limits, human approval, provenance logging | PilotProxy and operations | Medium |
| Pathfinder commissioning runs | Telescope | Large in calendar time |

## Limits

- It does not pay off until the relay carries the pilot statistic to the sister channels (dissertation
  App. G). Today the mask reaches only the pilot coarse channel, which is not used for science.
- It re-selects thresholds on the coupling and the pilot-to-shelf conversion that the bundle carries. It cannot
  re-measure those; that still needs a dedicated measurement.
