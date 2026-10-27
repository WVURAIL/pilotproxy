# Same-frame capture characterization

The purpose of `tools/characterize_capture_transfer.py` is to freeze and run a
descriptive study of the four simultaneous captures of September 16 and 17,
2026. The model is fit on the first two epochs, and the last two are held out
from that fit. Since the captures were examined before this study, they do not
form a blinded validation sample.

The protocol is written before any new products are measured. It binds the
candidate profiles, the original weight bank, the shared library, the input
serial maps, the native metadata, and the common available payload cells. Each
later operation refuses to overwrite its outputs and saves an input/output
receipt. After an interruption, a file-level operation resumes without
restarting completed files.

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

Run `reduce-one` for every protocol channel and epoch, CH37 as well, before
summarizing. The eight finite channel words are copied from the candidate
without retuning. Channels 17 and 35 are permanently rejected and are reduced
as diagnostics. Their unmasked measurements are descriptive, and their
operational retained set is empty. CH37 has no policy word and is kept as an
unmasked control. These operations leave the review-only candidate inactive
and build no production configuration.

## Measured quantities

The raw high and low nibbles are decoded as the real and imaginary
offset-binary components and then ordered by input ID. Each exact saved FPGA
origin defines one 16384-sample frame. The detector applies its Nyquist-zone
convention (zone 2, mirrored) in 128-sample windows. The per-input frame powers
(the all-bin sum of the fixed-point fine powers of each input) must reproduce
the global frame powers exactly, and the raw per-input autos must reproduce the
saved pilot-cell autos exactly. Input serial maps must match over the native
pilot, control, and payload captures. The fine decision uses all 2048 inputs
and the exact frozen channel word.

The signed pilot coordinate is
`x = SHELF * (P_target / (P_lower + P_upper) * ref_norm / target_norm - 1)`.
The same formula is applied to the global and the per-input frame powers. A
global power ratio and an equal-input mean of ratios are different quantities.
When the reference power is zero, the coordinate is recorded as missing, and no
negative estimate is clipped.

For each input, the response is formed from the mean raw auto power over the
common native-backed nonpilot cells, divided by the same-frame auto of CH37
(fid 491). This ratio is divided by its weighted training mean, and 1 is
subtracted. An intercept and a slope are fit against the signed per-input
pilot coordinate, using only the training rows. Every original epoch has equal
total weight, and every original frame of an epoch has equal weight. Masking
keeps those original weights. The live inputs are fixed from the training
autos, control powers, and finite predictors alone.

For each epoch, the held-out error is compared with the frozen training-mean
baseline, over all frames and over the frames retained by the current word.
The declared descriptive criterion for the retained frames is a mean squared
error no larger than the baseline in each held-out epoch. The all-frame
result is reported on its own. This test checks a conditional differential
prediction. It cannot determine the absolute DTV power or
distinguish between changing sky, receiver leveling, gain, and interference.
Daily gain solutions are available but are not applied in this raw-power
test. Missing native payload support, no selected frames, or nonfinite
fixed input coordinates are reported as unavailable.

## Temporal scope

The signed complex raw cross moments and the covariance, demeaned in each
capture, are saved at lags 0 to 8, up to 335.54 ms. Only pairs at exact FPGA
spacing with both endpoints retained contribute. Pairs stay in one capture and
use the actual FPGA separation, so missing frames are never compressed into
adjacent times. The representative payload cell is the one closest in
frequency to the channel midpoint, and it is chosen before any power is
measured. Its saved raw redundant east-west co-polarized classes (1,0), (2,0),
and (3,0), with distinct within-cylinder baselines of at least 40 m, are
normalized by 16384 times the same-frame mean auto power of the representative
cell over the fixed training inputs. The raw moment is the uncentered moment of
this normalized visibility. No gain is applied after stacking. The signed
off-diagonal coherent cross moment is kept without a square root or positive
clipping.

Each half, fit and evaluation, has only two acquisitions. Inputs, redundant
classes, frames, and lag pairs are therefore not independent validation
replicates, and the tool reports descriptive moments and predictions with no
confidence interval. The existing support floor of eight acquisitions is
recorded to explain why the block uncertainty summary is unavailable. Reaching
that count would not resolve physical identifiability. A long-timescale
coherence gain or the effective integration factor of the Paper II forecast
cannot be determined from these short finite records.

The analyzed frames are aligned to the capture. Exact FPGA origins are the only
matching keys, so these frames must never be joined by row index to the
global-origin Kotekan replay or used to qualify pair-expanded production
selection.

Raw files and large products are stored externally. The receipts hash the
selected raw frame bytes, the saved arrays that were used, and the exact
products. When only selected frames were read, no whole-file HDF5 hash is
recorded.

The completed study ran at commit `fbc548d2e1cf79cc3ef89739a79b4c65b32ce103` with
the source amendment `source_amendment_v3.json` (per-frame checkpoints, verified
per-epoch and per-payload-cell summary caches, and stricter input-byte identity
checks). To resume or replay it, use that source and pass
`--source-amendment /path/to/source_amendment_v3.json` to `reduce-one` or
`summarize`. A new execution freezes the current source. Its executed sources and
product hashes are kept with the private evidence.

Frame checkpoints bind the selected raw bytes, FPGA origin, protocol, executing
source hash, and a digest of every numerical array and identity field. A
checkpoint that mismatches, or that lacks this digest, is refused. The
per-epoch caches store the verified observations and receipts needed by the
fit. Cache reuse requires an approved source hash and an unchanged protocol and
payload digest.
