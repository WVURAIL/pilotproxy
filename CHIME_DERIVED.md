# CHIME-derived material

Some files in this repository hold results derived from CHIME data: the CHIME/FRB
baseband archive, the 2026 matched baseband capture, and the products and releases
built from them. They are aggregates only. No file here holds voltages, visibilities,
gain values or a listing of archive events. They are kept because the dissertation and
its papers reproduce from them.

Their public release needs CHIME collaboration clearance under its publication policy.
Until that clearance is given, a public copy of this repository must be the code-only
cut described at the end of this page. Material that is not CHIME data (the synthetic
GNU Radio benches, the LimeSDR lab runs, the designed weight banks, RFIsher's forecast tolerance table) is not listed.

## Data files

| Path | What it holds |
|---|---|
| `docs/evidence/ca_os_archive_bulk_2026-09-29/` (65 files) | Archive-bulk statistics of all 23 ATSC channels: per-channel bulk statistics, persistent-bin profiles and geometry (`data/bulk/chNN.json`), the per-channel release ledger with product SHA-256, era months and frame counts, the producer and verifier code, and the report |
| `docs/evidence/ch37_rank_injection_2026-09-30/` (143 files) | Channel 37 control study: rank curves and real-null exceedance rates of the ch37 control product (`data/controls_summary.json`), per-geometry frame metadata, beside a synthetic rank bench; with the result, check and verifier |
| `docs/evidence/youden_j_2026-09-07/` (2 files) | Coarse-versus-fine ROC and Youden J on the released per-pilot products: channel 35 off-era frames as the null, channels 34 to 36 from 2025 as the signal |
| `docs/evidence/F1_INVENTORY.md` | Index of the two packages above (file counts, bytes, manifest SHA-256) |
| `data/provenance/combine_subset_20260714/` (3 files) | Stack-subset decision record: the count of archive events per freq_id presence signature (190 signatures with counts, no event numbers) and the selected 16-channel block |
| `data/provenance/dissertation_summary_v1.json`, `_v2.json`, `_v3.json` | Curated per-channel and per-epoch values of the archive trawl (mask fractions, retained frames, status) |
| `tests/data/threshold_decisions_2026-10-12.json` | Threshold decisions per channel: deployed word or reason not deployed, shelf, exclusions and historical conversion inputs; a test golden for the deployment compiler |

## Inline quotes

These files are code or documentation, but they quote CHIME-derived numbers inline.
They stay in both cuts. Review them before any public release:

- `docs/ARCHIVE_HEALTH_REPAIR.md`, `docs/CANFAR_RUNBOOK.md` and
  the archived inventory records: archive accounting (units, events,
  exclusions, frame counts).
- `docs/DESIGN_DECISIONS.md`: archive measurements behind design choices (for example
  the coarse flag's false-alarm rate on quiet frames).
- `docs/PER_PILOT_PRODUCT_FIELDS.md`: railed-sample counts measured on real CHIME baseband
  (freq_id 506).
- Test goldens written inline: verified off dates
  (test_transmitter_off.py) and era literals (tests/config/test_profile_literals.py).
  Tests that need the withheld files skip in the code-only cut, but their source
  still holds these values.

## Era lists

The era lists are held by the WVU RAIL group and are not in this tree. The two files the
project profile reads, `projects/chime_atsc/eras/author_eras_2026-09-23.json` and
`transmitter_off.json`, are empty stand-ins, so tests of era values skip. The detector,
bundle export, `calibrate-bundle` and `test-bundle` do not use them. To repeat the archive
characterization, copy the real lists over the stand-ins.

## Code-only cut

`python tools/make_code_only_cut.py OUTPUT_DIR` (or `make code-only-cut OUT=OUTPUT_DIR`)
copies the tree without the paths below. The project profile needs an era list, so the
cut keeps the empty stand-ins for `author_eras_2026-09-23.json` and `transmitter_off.json`
(no dated channel, no off interval). Tests that read a withheld file skip with the reason
"withheld from the code-only cut".

```
docs/evidence/ca_os_archive_bulk_2026-09-29/
docs/evidence/ch37_rank_injection_2026-09-30/
docs/evidence/youden_j_2026-09-07/
docs/evidence/F1_INVENTORY.md
data/provenance/combine_subset_20260714/
data/provenance/dissertation_summary_v1.json
data/provenance/dissertation_summary_v2.json
data/provenance/dissertation_summary_v3.json
tests/data/threshold_decisions_2026-10-12.json
```
