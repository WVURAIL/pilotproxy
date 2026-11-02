# Detector-to-threshold workflow

1. Reduce the archive with the fixed detector and receiver configuration.
2. Select one declared receiver-state interval and retain its original event weights.
3. Export exact fine-score words and frame powers.
4. Build a cache with: pilot-proxy characterize score-cache --metadata
   PATH/ch19.json --output PATH/cache --current-word WORD
5. Generate an RFIsher schema-v3 additive-power response table.
6. Run python tools/threshold_rule.py --scores PATH/cache --table PATH/channel_table.json
   --channels 19 --power-limits PATH/finite_limits.json --out PATH/decisions
7. Evaluate the selected word on independent data, then compile and test the
   receiver configuration before using its mask.

The channel table supplies the coefficient in a = power_scale*x**2.
The simple model assigns temperature Tsys*x and spreads its squared amplitude
uniformly over supported forecast modes. This stated normalization is not an
empirical sky calibration.

The sweep retains original event weights, averages signed excess before
flooring, and carries an event-clustered sampling error. It chooses maximum
retention subject to the upper first-order bias and checked finite-power
limit. finite_limits.json supplies those per-channel limits in mK^2.
Channel 37 remains an unmasked background control.

The two-standard-error margin is an in-sample rule, not a simultaneous
confidence guarantee. Keep calibration and evaluation populations distinct.
Runtime correctness requires exact Q16 words, matching weights and receiver
geometry, frame alignment, and valid-sample accounting.
