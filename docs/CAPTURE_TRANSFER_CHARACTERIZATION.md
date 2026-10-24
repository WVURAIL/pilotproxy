# Same-frame capture characterization

`tools/characterize_capture_transfer.py` freezes and runs a descriptive comparison
on the four September 16–17, 2026 simultaneous captures. The first two epochs fit
the model; the last two are held out from that fit. The captures were examined
before this study, so they are not a blinded validation sample.

The protocol is written before any new measured products. It binds the candidate
profiles, original weight bank, shared library, input serial maps, native metadata,
and common available payload cells. Each subsequent operation refuses overwritten
outputs and saves an input/output receipt. A file-level operation can be resumed
after interruption without restarting completed files.

```sh
export PYTHONPATH=/path/to/pilotproxy/src
python tools/characterize_capture_transfer.py freeze \
  --datasets /path/to/datasets --bundle /path/to/review-only/bundle \
  --weight-bank /path/to/original/ch14-to-37.bin \
  --library /path/to/K128/libpilotproxy.so --output /path/to/output
python tools/characterize_capture_transfer.py reduce-one \
  --protocol /path/to/output/protocol.json --channel 29 --epoch 20260916162300
python tools/characterize_capture_transfer.py summarize \
  --protocol /path/to/output/protocol.json --channel 29
```

Run `reduce-one` for each protocol channel and epoch, including CH37, before
summarizing. The eight finite channel words are copied from the candidate without
retuning. Channels 17/35 are separate permanently rejected diagnostics; their
unmasked measurements are descriptive and their operational retained set is empty.
CH37 has no policy word and is kept as an unmasked control. None of these operations
activates the review-only candidate or constructs a production configuration.

## Measured quantities

The raw high/low nibbles are decoded as real/imaginary offset-binary components,
then ordered by input ID. Each exact saved FPGA origin defines one 16384-sample
frame. The detector applies its Nyquist-zone convention (zone 2, mirrored) in
128-sample windows. The per-input frame powers (each input's all-bin sum of its
own fixed-point fine powers) must reproduce the global frame powers exactly;
the raw per-input autos must reproduce the saved pilot-cell autos exactly. Input
serial maps must match across native pilot/control/payload captures. The fine
decision uses all 2048 inputs and the exact frozen channel word.

The signed pilot coordinate is
`x = SHELF * (P_target / (P_lower + P_upper) * ref_norm / target_norm - 1)`.
This same formula is applied to global and per-input frame powers; global power
ratios and an equal-input mean of ratios are distinct quantities. Zero reference
power is missing, never zero. No negative estimate is clipped.

For each input, the response starts with the mean raw auto power over the common
native-backed nonpilot cells, divided by the same-frame CH37 fid 491 auto. Divide
this ratio by its training weighted mean and subtract 1. Fit an intercept and slope
against the signed per-input pilot coordinate, using only training rows. Every
original epoch has equal total weight and every original frame within it has equal
weight. Masking preserves those original weights. Live inputs are fixed using only
the training autos, control powers and finite predictors.

Held-out error is compared with the frozen training-mean baseline for each epoch,
on both all frames and the current-word retained frames. The declared descriptive
criterion for the retained view is no larger mean squared error in each held-out
epoch; the all-frame comparison is reported separately. This tests a
conditional differential prediction; it cannot identify absolute DTV power or
distinguish changing sky, receiver leveling, gain, and interference. Daily gain
solutions are available but are not applied in this raw-power comparison. Missing
native payload support, no selected frames, or nonfinite fixed input coordinates
are reported as unavailable.

## Temporal scope

Signed complex raw cross moments and within-capture demeaned covariance are saved
at lags 0–8, up to 335.54 ms. Only exact FPGA-spaced pairs with both endpoints retained
contribute. Pairs remain within captures and use actual FPGA separation; missing
frames are not compressed into adjacent times. The representative payload
cell is chosen by frequency proximity to the channel midpoint, before measuring
power. Its saved raw redundant east-west co-polarized classes (1,0), (2,0), and (3,0), with separate within-cylinder baselines of at least
40 m are normalized by 16384 times the same-frame representative-cell mean auto
power over the fixed training inputs. The raw moment is the uncentered moment of
this normalized visibility. No gain is applied after stacking. The signed off-diagonal coherent cross
moment is retained without square root or positive clipping.

There are only two acquisitions per fit/evaluation half. Inputs, redundant classes,
frames and lag pairs are not independent validation replicates. This tool reports
descriptive moments and predictions, with no confidence interval. The existing
eight-acquisition support floor is recorded only to explain the unavailable block
uncertainty summary; reaching that count would not resolve physical identifiability.
The short finite records do not determine a long-timescale coherence gain or the
Paper II forecast's effective integration factor.

The analyzed frames are capture-aligned. They must not be joined by row index to
the separate global-origin Kotekan replay or used to qualify pair-expanded
production selection. Exact FPGA origins are the only matching keys.

Raw files and large products remain external. Receipts hash selected raw frame
bytes, the used saved arrays, and exact products; they do not claim a whole-HDF5
byte hash when only selected frames were read.

The completed study ran at commit `fbc548d2e1cf79cc3ef89739a79b4c65b32ce103` with
the source amendment `source_amendment_v3.json` (per-frame checkpoints, verified
per-epoch and per-payload-cell summary caches, and stricter input-byte identity
checks). To resume or replay it, use that source and pass
`--source-amendment /path/to/source_amendment_v3.json` to `reduce-one` or
`summarize`. A new execution freezes the current source. Its executed sources and
product hashes are kept with the private evidence.

Frame checkpoints bind the selected raw bytes, FPGA origin, protocol, executing
source hash, and a digest of every numerical array and identity field; a
mismatched checkpoint or one without that digest is refused. The per-epoch caches
keep the verified observations and receipts the fit needs. Cache reuse requires an
approved source hash and an unchanged protocol and payload digest.
