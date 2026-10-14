# Weight banks

The files directly under this directory are the active, profile-backed banks:

- `chime_dtv_weights_k128.bin` and its manifest select the verified
  DC-centered CHIME baseband convention;
- `chord_dtv_weights_k64.bin` and its manifest select the CHORD geometry.

Treat each binary and adjacent manifest as one immutable pair. Runtime loading
checks the binary digest recorded by the manifest; documentation builds derive
their displayed digest prefixes from these active files.

## Historical CHIME bank

The bank of the 2026-07 manuscript sweeps and the first archive epoch assumed
the earlier half-band baseband convention and must not be selected for a
current detector run. It left the repository; it is
`weights/legacy_halfband/chime_dtv_weights_k128.bin@b52032c`, with its
manifest beside it at that commit.

The manuscript was removed from this repository; its provenance hash list is
`paper/manuscript/provenance/hashes.sha256@30e50c2`, checked with `sha256sum
--check --ignore-missing` from a worktree at that commit. That commit, like
the `@b52032c` above, is in the private provenance repository
(see `PROVENANCE.md`).
