# Single-input SDR digital adapter

The purpose of `pilot_proxy.integration.sdr_upgrade_adapter` is to prepare an
existing contiguous complex IQ record for the upgrade detector geometry on the
CPU. The adapter has no device or GPU API. It matches the 390625 Hz output rate,
the 16384-sample frame, K=128 samples per projection, and L=128 rows per input,
and it supplies one input. Its FIR is a declared engineering filter. It does not
emulate the CHIME analog chain or polyphase filter bank, and it carries no
measured calibration of either.

## Digital contract

The caller must provide the nominal input rate of 2000000 Hz, a signed digital
mixer frequency, an integer target bin, and a positive fixed sample quantization
scale. A positive digital tone is `exp(+2j*pi*f*n/input_rate)`. Mixing
multiplies the IQ by `exp(-2j*pi*mixer_hz*n/input_rate)`, so the output digital
frequency is the input digital frequency minus the mixer frequency. Neither the
API nor its receipt assigns a physical RF frequency. An LO readback alone
determines neither the spectral orientation nor an RF calibration.

The anti-alias stage interpolates by 25 and decimates by 128. Its FIR has 8193
real coefficients, a Kaiser window with beta 8.6, and a 165 kHz cutoff at the 50
MHz interpolated rate. With unity gain at DC, the coefficients are generated
explicitly with `scipy.signal.firwin`, and `upfirdn` applies their interpolation
gain of 25. Every coefficient is saved in the receipt, with the SHA-256 of their
canonical little-endian float64 bytes. For numerical acceptance, the amplitude
error must be below 5e-5 up to 135 kHz, and the amplitude must be below 4e-5
from the output Nyquist frequency (195312.5 Hz) up to 25 MHz. The spacing of the
frequency grid is recorded. Only the digital transfer is covered by these
checks, and no bound is set on the analog receiver response.

The three natural projectors are the target, the lower digital reference, and
the upper digital reference, at bins `[b, b-2, b+2]` with a bin width of
3051.7578125 Hz. All three centers must lie in the qualified +/-135 kHz
passband. Each natural weight has positive phase, and the projection is
`x dot conjugate(w)`. A translated band that crosses the Nyquist boundary of the
raw input is refused by the configuration, and an unknown Nyquist zone is
neither inferred nor reversed.

Samples and weights use the symmetric signed integer rails [-7,7], nearest
rounding with ties to even, and packed two's-complement components, with the
real part in the high nibble and the imaginary part in the low nibble. The
weight scale is 7, while the sample scale is explicit, fixed by the caller, and
never fitted to each frame. Receipts count the components that require clipping
and save the weight norms. Integer row projections use the existing CPU kernel
reference. The frame powers (`term_power_sums`, exact uint64) follow the
kernel's definition. For each term, the 128 row projections are passed through
the frozen `fxfft256`, and the fine powers are summed over all 256 bins (receipt
schema `sdr-upgrade-digital-adapter-v2`, while v1 receipts hold the sum of
`|z|^2` over the rows). From those powers, the reported coarse ratio is exactly
`2*P_target/(P_lower + P_upper)`, with no weight-norm renormalization. A zero
denominator is reported as infinite or undefined and never becomes a noise
observation.

## Timing and framing

For the untrimmed resampler output index `j`, the sample-center time is
`(128*j - 4096)/50000000` seconds relative to the start of the input record.
Only the samples with complete FIR support in the record are kept, and the
first retained sample is at 81.92 microseconds. Only complete 16384-sample
frames are processed, and the unused final samples are counted. The frame
duration is 0.04194304 seconds, and the row spacing is 0.00032768 seconds. The
end of the final frame is the next sample center, exclusive. Device clock readback and
absolute UTC calibration are not part of this nominal digital time lattice.

This adapter works on one contiguous record. Calling it independently on
successive chunks loses the FIR state and drops the boundaries. A live stream
needs stateful FIR history, mixer phase and output-lattice state, and explicit
handling of transport gaps before it can feed a live detector, and these are
provided by the streaming adapter in [SDR_UPGRADE_STREAMING.md](SDR_UPGRADE_STREAMING.md).

For this one-input geometry, an ideal IID Gaussian input with no filtering, no
quantization, and equal norms would give a coarse frame ratio distributed as
F(256,512), and the fixed-point transform departs from it only by its rounding.
Filtering, quantized weights, rounding, clipping, and measured receiver behavior
each require their own validation. Copying or reweighting one SDR stream does
not turn it into 2048 independent CHIME inputs.

## Reproduce the digital validation

The digital validation needs the repository test extra (NumPy, SciPy, and
pytest) and is run with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/core/test_sdr_upgrade_adapter.py -q

python tools/validate_sdr_upgrade_adapter.py \
  --capture-root /path/to/sdr_reference_transport_2026-09-09 \
  --output /path/to/new-output
```

The second command only reads the accepted RX-only record taken with the antenna
attached. Both the input and its receipt are verified against their release
manifest and the frozen hash of the accepted record. The tool sets the thread
limit of each numerical library to one, and it refuses a
nonempty output directory. It saves the synthetic signed-tone and
alias-rejection checks, an exact independent scalar check of the integer
projection for every replay row, the per-frame powers and timing, the filter
coefficients, the weights, source snapshots, and a hash manifest. In the
synthetic-tone check, the desired-term power is divided by the larger of one
squared projection-code unit and the largest other-term power. In the receipt,
this ratio is labeled as a lower bound with that floor on the denominator, and
the actual other-term power is recorded beside it. When both other powers are
zero, the reported finite value differs from the exact ratio of desired to
other power.

The ambient replay uses a declared mixer of +100 kHz, target bin zero, and a
sample scale of 500. Its references are therefore at +/-6103.515625 Hz at the
output, which corresponds to the input digital frequencies 93896.484375 and
106103.515625 Hz. In the 2 MHz, K=128 engineering project, the references are at
68750 and 131250 Hz, so results from the two analyses cannot be exchanged. No
fitted tones are subtracted in the replay, its scale is not optimized, and the
antenna record is not labeled as thermal noise. Its distribution is a
descriptive engineering observation with a single-input denominator.

## Next controlled-capture acceptance

A claim of physical calibration, or a test of qualified noise against
stable-signal distributions, needs the following steps on this path:

1. Determine the physical frequency sign and offset with independently known
   positive and negative offsets. Record the LO and sample-clock readbacks with
   their uncertainties, the IQ ordering, the gain, the antenna or termination,
   the bandwidth, the complete capture interval, and the transport receipt.
2. Measure the analog and digital transfer together over the target and
   reference supports, with the leakage, image response, and edge behavior.
   Freeze the filter coefficients, projector weights, gain, and quantization
   scale from a dedicated calibration record. Check the clipping and branch
   covariance on independent evaluation records.
3. Collect terminated or independently bounded noise, then a stable signal plus
   that noise at predeclared levels, with the hardware state settled and enough
   independent frames for the chosen confidence interval. A transmitter-off
   record with the antenna attached is not a thermal null by itself.
4. Evaluate these single-input data with single-input degrees of freedom, never
   with those of the array. Report the analog gain calibration and the physical
   signal-to-residual transfer apart from the integer arithmetic agreement.
5. Validate the stateful streaming and the handoff to the production fine
   detector. Parity of the CPU row projections certifies neither the fine policy
   nor the Pathfinder.

The current digital receipt closes the preparation and arithmetic checks only,
and the physical calibration and live Pathfinder tasks remain open.
