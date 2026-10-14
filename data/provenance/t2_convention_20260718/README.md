# T2 estimator-convention and sample-span provenance (2026-07-18)

`T2` is the project label for the second technical review item: whether the
reported ATSC pilot-to-data ratio changes with the estimator convention or
with the sample span. We keep the two effects separate because either one can
move the calibration number without changing the detector.

The audit and trim reports named below are now retained. T2 remains open
because their input IQ captures, file hashes, exact commands, and the
stationary-span sweep bundle are not all present in this repository; numbers
that depend on this calibration therefore remain provisional.

## Artifacts committed in this directory

- `audit_orig_fullspan.json` is the extended audit of the original capture at
  `--max-samples 600000`. It uses 524288 samples, or 8 complete segments, and
  reports `measured_pilot_below_data_direct_db = 11.5950992961` dB.
- `audit_trim_fullspan.json` applies the same audit to the trimmed capture. It
  uses 524288 samples and reports
  `measured_pilot_below_data_direct_db = 11.0828912346` dB.
- `block_profile_20260718.txt` records the ten-block coherent projection that
  identifies the generator startup transient.
- `audit_v3.json` is the default-span audit of the trimmed capture and reports
  a direct result of 11.4969224266 dB.
- `audit_v4.json` is the default-span audit of the stationary trimmed capture
  and reports a direct result of 11.4044465569 dB.
- `trim_report.json` and `trim_report_stationary.json` record the corresponding
  pilot-amplitude adjustments.

## Recorded 2-by-2 comparison

The default-span direct audits use 262144 samples and record 12.024 dB for the
original capture and 11.497 dB for the trimmed capture. Together with the
full-span records, the comparison gives:

- sample-span changes of `-0.43/-0.41` dB at fixed method;
- method-versus-projection differences of `-0.24/-0.22` dB at the long span;
- trim shifts of `-0.527/-0.512/-0.530` dB across the three recorded
  conventions; and
- data power equal to nine significant digits between the two full-span
  captures.

For the first two bullets, each pair is ordered original/trimmed. The three
trim-shift values are ordered default-span direct/full-span direct/projection.
These values are the recorded decomposition. The JSON results are present, but
the decomposition is not independently reproducible from a bare clone because
the source IQ captures and complete command/hash ledger are not committed.

## Status of the remaining evidence

- **Retained here:** `trim_report.json` was regenerated after the `gain_db`
  formula correction. It records
  amplitude factor `1.0629299002401351` and power gain
  `0.5300924787490547` dB. The earlier value 0.265 dB was a reporting-formula
  error; the trimmed capture itself was not the error.
- **Retained here:** `trim_report_stationary.json` and `audit_v4.json` document
  the cropped/retrimmed audit result.
- **Still external:** the original, trimmed, cropped, and stationary IQ bytes;
  their cryptographic identities; the exact crop/retrim/audit commands; and the
  stationary-span sweep output required by the G3b protocol.

T2 closes only after the input identities, exact commands, and stationary-span
sweep are retained together.
