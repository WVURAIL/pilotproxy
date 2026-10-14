# Fine-gain Monte Carlo — 2026-08-19

Measured coherent-gain evidence for the fine designated-set reduction,
produced by `tools/measure_fine_gain.py` at the deployed geometry
(2048 streams x 128 windows), integer row sums, deployed statistics,
empirical H0 thresholds at fixed Pfa. The batched reduction is verified
against `pilot_proxy.fine_reduction.fine_reduce` (`--stage verify`, and
`tests/core/test_measure_fine_gain.py`).

Reproduce:

    python3 tools/measure_fine_gain.py --stage verify --trials 4 --out OUT
    for s in 1 2 3; do
      python3 tools/measure_fine_gain.py --stage h0 --trials 4000 --seed $s --out OUT
    done
    for snr in -40 -37 -34 -31 -28 -25 -22 -19 -16 -13 -10; do
      python3 tools/measure_fine_gain.py --stage sweep --snr-db $snr --trials 1200 --seed 7 --out OUT
    done
    for snr in -34 -31 -28 -25 -22; do
      python3 tools/measure_fine_gain.py --stage sweep --snr-db $snr --trials 1200 --seed 7 --half-bin --out OUT
    done
    python3 tools/measure_fine_gain.py --stage report --out OUT

Result (12,000 H0 trials; 1,200 trials per sweep point):

    Pfa=0.01:  SNR@Pd=0.5 coarse -22.90 dB, fine -32.22 dB -> gain 9.32 dB
    Pfa=0.001: SNR@Pd=0.5 coarse -21.04 dB, fine -30.81 dB -> gain 9.77 dB

consistent with the per-channel 9.4-10.0 dB measurement the analysis
books as 10 dB of fine-stage credit.
