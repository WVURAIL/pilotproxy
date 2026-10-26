# Power-era proposals

The purpose of `python -m pilot_proxy.characterization.power_eras` is to
propose changes in the dominant relative spectral feature of a channel. The era
list used by the paper comes from `characterize era-boundaries --line-identity`
(`projects/chime_atsc/eras/line_identity_2026-10-17/`, see START_HERE.md,
step 2). The proposals of this tool are recorded in
`projects/chime_atsc/eras/power_eras_2026-10-04.json`. Accepted boundaries,
detector calibration, and science tolerance are set elsewhere. All real channel
displays were inspected before this design, and none is held out. The defaults
are practical choices and carry no calibrated false-alarm guarantee.

## Input and command

Supply one channel per JSON document:

```json
{
  "schema_version": "monthly-power-features-v1",
  "channel": 19,
  "provenance": {
    "source_sha256": "the source artifact digest",
    "reduction": "frame maximum -> event median -> monthly median"
  },
  "months": [
    {
      "month": "2024-01",
      "level_db": -0.4,
      "event_count": 8,
      "frame_count": 40,
      "day_count": 4,
      "peak_offset_hz": 12.0,
      "tracking_usable": true,
      "metadata": {}
    }
  ]
}
```

```console
python -m pilot_proxy.characterization.power_eras --input monthly_features.json --output NEW.json
```

The output must not already exist. The command records the exact input-file
SHA256. The pure API is
`propose_power_eras(months, channel=19, provenance=..., config=PowerEraConfig())`,
where each row is a `MonthlyPower`. It records canonical digests of the input
and the configuration. Provenance is preserved as supplied, and this module
does not authenticate it.

The input is the unsmoothed monthly median of event medians of frame maxima,
in one fixed spectral window. Each source event has equal weight. The caller
must authenticate that reduction and the source-event identity, since neither
can be recovered from an already reduced scalar. Duplicate or unordered months,
unknown fields, nonfinite numbers, duplicate JSON keys, and invalid types are
rejected. A missing measurement is an explicit `null`, and a negative relative
level in dB is valid.

A supported month has finite power, usable tracking, at least 5 events,
30 frames, and 3 days. Missing day counts fail the default gate with
`day_support_unverified`, and setting `min_days=0` disables that gate
explicitly. Optional `config` fields override named numeric defaults and change
the digest. The five-month blocks and the four consistent months are fixed in
the structure of this version and cannot be configured.

## Candidate rule

At each possible edge, the five supported calendar months immediately before
the edge are compared with the five immediately after it. Months are never
compressed across a gap. The center of each block is its median, D is the
absolute separation of the two centers, and the robust scale of each block is
1.4826 times its median absolute deviation.

- The scale is the larger block scale or 0.25 dB, whichever is larger.
- The sign of the center difference is kept. The smallest signed separation is
  found over the two original medians and over all medians with one month
  removed from either block or from both. It must be at least
  `max(3 dB, 3 * scale)`, inclusive.
- In each block, at least four values must be no more than
  `min(D/3, max(1 dB, 2 * block_scale))` from the median of that block.
- Adjacent passing edges are clustered. The representative edge has the
  largest ratio of separation to required separation, and the earliest edge
  breaks ties. Every edge and the complete candidate interval are preserved.

The 3 dB setting is a practical separation in relative power and has no
relation to the RFIsher tolerance of 1. Under the default configuration,
smaller changes are missed by design.

The observation bounds come from the consistent samples of each block, refined
only by a contiguous old-consistent prefix of the right block or a
new-consistent suffix of the left block. Interior outliers at the opposite
level cannot reverse the bounds. The uncertainty of the candidate edge is
recorded on its own. Confirmation requires five actual adjacent supported
months starting at the first observed new month, so an archive endpoint or an
intervening gap cannot borrow confirmation from later months.

Across a missing or unsupported run, the last five consecutive supported months
before the run are compared with the first five after it. A supported
difference is reported as `gap_censored_power_change`, with no exact transition
time in the gap. Compatible observations on the two sides do not show
continuity, and a gap alone never creates a boundary.

## Output and interpretation

`boundaries` contains reviewable clusters, their representative diagnostics,
observation bounds, transition-zone months, feature diagnostics, and any
overlapping confirmation windows. `proposed_cohort_first_month` is an observed
bound on the onset, and the exact date of the transmitter change is not
inferred. `pending_changes` holds changes without enough consecutive
support. `edge_checks` records the rejected comparisons and their reasons.
`unsupported_ranges` and `gaps` record missing coverage.

`excursions` is a display diagnostic. An excursion is a month that differs from
the median of its five-month flanks by at least the configured practical and
robust requirement, while the two flank medians differ by less than the
practical threshold. Excursions never create a boundary. The raw monthly
records are always kept.

The module produces candidate clusters. An accepted or complete working
partition is left to the consumer, which may build a provisional partition from
unambiguous candidates after resolving overlap and timing explicitly. Such a
partition must carry its own label and must preserve these diagnostics and the
accepted era list.

In the synthetic suite, pulses of one, two, and three months fail the
lasting-regime tests. Episodes of four months and seasonal changes can still
produce candidates. A linear ramp can fail the consistency test or leave an
uncertain transition, and ramps in general are not guaranteed to be rejected.
The event-count gates set the required support and say nothing about
statistical independence.

Frequency offsets never trigger boundaries, and a constant-power feature that
drifts in the window should remain one power regime. A feature leaving the
window, a secondary tone, DC, an instrument change, or another feature becoming
dominant can mimic a power change. Offsets and metadata are diagnostic, and the
spectrogram should be inspected. A low relative level or a confirmed power
change says nothing about the absence of RFI, the identity of the transmitter,
or the physical residual tolerance.
