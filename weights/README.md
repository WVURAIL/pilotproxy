# Weight banks

This directory holds the detector weight banks shipped with the package. Each
bank is a binary and its adjacent manifest, built from a receiver profile in
`configs/receiver_profiles/`.

- `chime_dtv_weights_k128.bin` is the CHIME bank (profile `chime_dtv_fengine`,
  128-sample detector window). It uses the DC-centered CHIME baseband
  convention, which was confirmed by the frame correction and the 23-channel
  parity census in `docs/VALIDATION_GATES.md`.
- `chord_dtv_weights_k64.bin` is the CHORD bank (profile `chord_dtv_fengine`,
  64-sample detector window) for the CHORD channelizer geometry of 8192 coarse
  channels.

Each binary and its manifest form one immutable pair. When a bank is loaded,
the SHA-256 digest of the binary is compared with `artifacts.weights_sha256` in
the manifest, and a mismatch is an error. The 16-character digest prefixes in
the data sheet are computed from the CHIME bank and its manifest by
`tools/generate_doc_specs.py`.
