# Power-era proposals

`python -m pilot_proxy.characterization.power_eras` proposes changes in a
channel's dominant **relative spectral feature**. It is the era procedure of
the threshold release of 2026-10-04, whose current-era spans it supplies:
channel 20 from 2024-07, channel 27 from 2022-11 and channel 34 from 2022-09,
the other channels unchanged. The section 8.1 rule
(`src/pilot_proxy/characterization/eras.py`) and the author-dated list of
2026-09-23 (`projects/chime_atsc/eras/author_eras_2026-09-23.json`) are the
earlier era rule; they are what the archive releases of 2026-09 used
(`releases.md`). The procedure does not set accepted boundaries, detector
calibration, or science tolerance. All real channel displays were inspected
before this design; none is held out. The defaults are practical choices, not
a calibrated false-alarm guarantee.

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
where each row is a `MonthlyPower`. It records canonical input and configuration
digests. Provenance is preserved, not independently authenticated by this module.

Use unsmoothed monthly medians of event medians of frame maxima in the same
fixed spectral window. Each source event has equal weight. The caller must
authenticate that reduction and source-event identity; this module cannot infer
them from an already reduced scalar. Duplicate or unordered months, unknown
fields, nonfinite numbers, duplicate JSON keys, and invalid types are rejected.
Missing measurements are explicit `null`; negative relative dB remains valid.

A supported month has finite power, usable tracking, at least 5 events,
30 frames, and 3 days. Missing day counts fail the default gate with
`day_support_unverified`; setting `min_days=0` explicitly disables that gate.
Optional `config` fields override named numeric defaults and change the digest.
This version fixes five-month blocks and four consistent months structurally.

## Candidate rule

At each possible edge, compare the immediately preceding and following five
supported calendar months. Months cannot be compressed across a gap.
Let the centers be their medians, their absolute separation be D, and their
robust scales be 1.4826 times median absolute deviation.

- The scale is the larger block scale or 0.25 dB, whichever is largest.
- Keep the sign of the center difference. Compute the smallest signed
  separation among both original medians and all medians after removing one
  month from either or both blocks. It must be at least
  `max(3 dB, 3 * scale)`, inclusively.
- In each block, at least four values must lie within
  `min(D/3, max(1 dB, 2 * block_scale))` of that block's median.
- Cluster adjacent passing edges. Choose the strongest separation-to-required
  separation ratio, with the earliest edge breaking ties. Preserve every edge
  and the complete candidate interval.

The 3 dB setting is a practical relative-power separation, unrelated to
RFIsher's tolerance of 1. It deliberately misses smaller changes under the
default configuration.

Observation bounds come from each block's own consistent samples, refined only
by a contiguous old-consistent prefix of the right block or new-consistent
suffix of the left block. Interior opposite-level outliers cannot reverse the
bounds. Candidate-edge uncertainty is recorded separately. Five actual
adjacent supported months starting at the observed first-new month are required;
an archive endpoint or intervening gap cannot borrow future confirmation.

Across a missing or unsupported run, compare the last and first five
consecutive supported months separately. A supported difference is
`gap_censored_power_change`, never an exact transition time inside the gap.
Compatible observations on either side do not establish continuity and do not
create a boundary simply because there was a gap.

## Output and interpretation

`boundaries` contains reviewable clusters, their representative diagnostics,
observation bounds, transition-zone months, feature diagnostics, and any
overlapping confirmation windows. `proposed_cohort_first_month` is only an
observed onset bound, not an inferred exact transmitter change date.
`pending_changes` preserves differences without sufficient consecutive support.
`edge_checks` retains rejected comparisons and reasons. `unsupported_ranges`
and `gaps` preserve missing coverage.

`excursions` is a separate display diagnostic: a month differs by at least
the configured practical/robust requirement from the median of its five-month
flanks, while those flank medians differ by less than the practical threshold.
It does not create a boundary. Raw monthly records are always retained.

The module produces candidate clusters, not an accepted or complete working
partition. A consumer may construct a separately labeled provisional partition
from unambiguous candidates after explicitly resolving overlap and timing;
it must preserve these diagnostics and the accepted historical record.

One-, two-, and three-month pulses fail the lasting-regime tests in the
synthetic suite. Four-month episodes and seasonal changes can still produce
candidates. Linear ramps can fail consistency or leave an uncertain transition;
there is no universal ramp-rejection claim. Event-count gates are support
choices, not proof of statistical independence.

Frequency offsets never trigger boundaries. A constant-power feature drifting
inside the window should remain one power regime. Window exit, a secondary
tone, DC, instrument changes, or another feature becoming dominant can mimic
a power change. Offsets and metadata remain diagnostic; inspect the spectrogram.
Neither a low relative level nor a confirmed power change certifies absence of
RFI, a particular transmitter identity, or physical residual tolerance.
