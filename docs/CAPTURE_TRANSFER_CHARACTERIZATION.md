# Same-frame capture characterization

`tools/characterize_capture_transfer.py` freezes and runs a descriptive comparison
on the four September16–17,2026 simultaneous captures. The first two epochs fit
the model; the last two are held out from that fit. These captures were previously
characterized, so this is not a blinded or previously unseen validation sample.

The protocol is written before any new measured products. It binds the candidate
profiles, original weight bank, shared library, input serial maps, native metadata,
and common available payload cells. Each subsequent operation refuses overwritten
outputs and saves an input/output receipt. A file-level operation can be resumed
after interruption without restarting completed files.

```sh
export PYTHONPATH=/home/djg/rail/pilotproxy/src
python tools/characterize_capture_transfer.py freeze \
  --datasets /path/to/datasets --bundle /path/to/review-only/bundle \
  --weight-bank /path/to/original/ch14-to-37.bin \
  --library /path/to/K128/libfstatistic.so --output /path/to/output
python tools/characterize_capture_transfer.py reduce-one \
  --protocol /path/to/output/protocol.json --channel 29 --epoch 20260916162300
python tools/characterize_capture_transfer.py summarize \
  --protocol /path/to/output/protocol.json --channel 29
```

Run `reduce-one` for each protocol channel and epoch, including CH37, before
summarizing. The eight finite channel words are copied from the candidate without
retuning. Channels17/35 are separate permanently rejected diagnostics; their
unmasked measurements are descriptive and their operational retained set is empty.
CH37 has no policy word and is kept as an unmasked control. None of these operations
activates the review-only candidate or constructs a production configuration.

## Measured quantities

The raw high/low nibbles are decoded as real/imaginary offset-binary components,
then ordered by input ID. Each exact saved FPGA origin defines one16384-sample
frame. The detector applies its inverted-spectrum convention in128-sample windows.
The per-input row-power sums must reproduce the global coarse integer sums exactly;
the raw per-input autos must reproduce the saved pilot-cell autos exactly. Input
serial maps must match across native pilot/control/payload captures. The fine
decision uses all2048 inputs and the exact frozen channel word.

The signed pilot coordinate is
`x = SHELF * (P_target / (P_lower + P_upper) * ref_norm / target_norm - 1)`.
This same formula is applied to global and per-input coarse powers; global power
ratios and an equal-input mean of ratios are distinct quantities. Zero reference
power is missing, never zero. No negative estimate is clipped.

For each input, the response starts with the mean raw auto power over the common
native-backed nonpilot cells, divided by the same-frame CH37 fid491 auto. Divide
this ratio by its training weighted mean and subtract1. Fit an intercept and slope
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
at lags0–8, up to335.54ms. Only exact FPGA-spaced pairs with both endpoints retained
contribute. Pairs remain within captures and use actual FPGA separation; missing
frames are not compressed into adjacent times. The representative payload
cell is chosen by frequency proximity to the channel midpoint, before measuring
power. Its saved raw redundant east-west co-polarized classes (1,0), (2,0), and (3,0), with separate within-cylinder baselines of at least
40m are normalized by16384 times the same-frame representative-cell mean auto
power over the fixed training inputs. The raw moment is the uncentered moment of
this normalized visibility. No gain is applied after stacking. The signed off-diagonal coherent cross
moment is retained without square root or positive clipping.

There are only two acquisitions per fit/evaluation half. Inputs, redundant classes,
frames and lag pairs are not independent validation replicates. This tool reports
descriptive moments and predictions, with no confidence interval. The existing
eight-acquisition support floor is recorded only to explain the unavailable block
uncertainty summary; reaching that count would not resolve physical identifiability.
The short finite records do not determine a long-timescale coherence gain or the
PaperII forecast's effective integration factor.

The analyzed frames are capture-aligned. They must not be joined by row index to
the separate global-origin Kotekan replay or used to qualify pair-expanded
production selection. Exact FPGA origins are the only matching keys.

Raw files and large products remain external. Receipts hash selected raw frame
bytes, the used saved arrays, and exact products; they do not claim a whole-HDF5
byte hash when only selected frames were read.

The October16 execution has explicit source-only amendments for per-frame
checkpoints, verified per-epoch and per-payload-cell summary caches, and stricter input-byte identity checks. The original protocol and
completed products are preserved. To resume that historical execution, use the
executed source at commit `fbc548d2e1cf79cc3ef89739a79b4c65b32ce103` and pass
`--source-amendment /path/to/source_amendment_v3.json` to `reduce-one` or
`summarize`. A new execution freezes the current source normally. Frame
checkpoints bind the exact selected raw bytes, FPGA origin, protocol, and executing
source hash; a mismatched checkpoint is refused. All executed source versions are
retained with the private evidence. These changes do not alter the numerical
analysis or shorten its declared capture population.

The per-epoch caches retain the verified observations and receipts needed by the fit. The final evidence retains these caches, but omits duplicate per-cell and per-frame checkpoints. Source revisions are explicit, and cache reuse requires an approved source hash plus unchanged protocol and payload digest.

After the study, frame checkpoints gained a digest of every numerical array and
identity field. New runs refuse altered payloads and legacy checkpoints without
that digest. The completed study retains its exact executed sources and product
hashes; this later guard does not change or requalify its numerical results.
