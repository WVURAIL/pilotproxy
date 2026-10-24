# Weight banks

The files directly under this directory are the active, profile-backed banks:

- `chime_dtv_weights_k128.bin` and its manifest select the verified
  DC-centered CHIME baseband convention;
- `chord_dtv_weights_k64.bin` and its manifest select the CHORD geometry.

Treat each binary and adjacent manifest as one immutable pair. Runtime loading
checks the binary digest recorded by the manifest; documentation builds derive
their displayed digest prefixes from these active files.
