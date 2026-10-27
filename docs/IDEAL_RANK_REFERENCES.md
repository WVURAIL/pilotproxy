# Ideal rank references

These studies compare reference statistics under independent fine-bin
F(4096, 8192) noise. Their thresholds are limited to that ideal model, so they
neither calibrate an operational `eta_Pfa` nor show that an archive cohort is
signal-free.

Run them with the analysis environment and a new output directory:

```sh
python -B -m pilot_proxy.testbench.rank_reference --out /path/to/rank-reference
python -B -m pilot_proxy.testbench.rank_quadrature --out /path/to/rank-quadrature --ranks 20 41 46 62 63 94
python -B -m pilot_proxy.testbench.cfar_reference --out /path/to/cfar-reference
```

`rank_reference` compares cell averaging with every order-statistic rank for
bulk sizes 125 and 126, with and without contaminated bulk bins. For
reproduction, the recorded seeds, 20 batches of 20000 frames, interpolation
grids, and direct root solves are kept. `rank_quadrature` checks the clean reference with the exact
order-statistic density and an FFT convolution for cell averaging.

`cfar_reference` compares cell averaging with ranks 63, 94, and 113. It keeps
the rectangular-window response, both anchor parities, common noise draws,
four million null frames per parity, and the recorded contamination cases.
The ranks compared here are study parameters. In operation, the detector uses
the fixed median rank.

The programs refuse an existing output directory. Each writes its numerical
JSON and a short scope note. Set the BLAS thread count to one when comparing
the outputs with the recorded reference runs.
