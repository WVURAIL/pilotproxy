# Corrected fine-gain reports, 2026-09-08

These reports re-read the existing integer row-sum Monte Carlo shards. No new
trials, voltage quantization experiment, or radio transmission were performed.
The old evidence directories and their reports retain their original bytes.

The old centered filename glob also matched `h1_half_*` files. The reporter now
uses each shard's saved `b0` and validates its stream count, noise scale, trial
shape, seed and transform identity. New shards save explicit transform metadata;
the older shards predate those fields, so their documented geometry is identified
as inferred in each source record rather than claimed to have been stored.
`gain_report.json` records every input path and SHA-256, population, trial count,
experiment identity, report-producer hash, thresholds and detection curves.

| Input evidence | Corrected report | Centered gain at Pfa=0.01 | At Pfa=0.001 |
|---|---|---:|---:|
| `m_scaling_2026-09-07/data/null_m2048` | `m2048/` | 9.39884 dB | 9.69459 dB |
| `fine_gain_mc_2026-08-19` | `august19_m2048/` | 9.43121 dB | 10.09393 dB |

Both retained samples contain 2048 streams and 12,000 null trials. The first
historical mixed-population report gave approximately 9.246/9.508 dB; its input
population, not the underlying detector arithmetic, caused that difference.
These are interpolated 50% detection crossings, not confidence bounds.

Reproduce from the pilot-proxy repository with the scientific Python environment:

```sh
python tools/measure_fine_gain.py --stage report \
  --out docs/evidence/m_scaling_2026-09-07/data/null_m2048 \
  --report-dir docs/evidence/fine_gain_report_correction_2026-09-08/m2048
python tools/measure_fine_gain.py --stage report \
  --out docs/evidence/fine_gain_mc_2026-08-19 \
  --report-dir docs/evidence/fine_gain_report_correction_2026-09-08/august19_m2048
```

The fine transform in this experiment uses floating arithmetic on integer row
sums. Full paired int4-input, integer-weight, fixed-transform fine-path sensitivity,
false-alarm tails and radio-path validation remain separate experimental work.
