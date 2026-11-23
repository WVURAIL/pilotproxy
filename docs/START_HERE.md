# Detector-to-threshold workflow

The words a bundle carries come from `calibrate-bundle` on a recording of the telescope where
they are used. [RUNNING_THE_DETECTOR.md](RUNNING_THE_DETECTOR.md) gives each step with its commands.

1. Record a detect-only run of the bound channels and the channel-37 control, several days
   long (Step 1).
2. Calibrate with `"threshold": "power_limit"`, or `plan-run` and `process-run` with
   `--threshold power_limit --power-limit inputs.json` (Step 2). The plan gives the
   `power_limit` inputs: the RFIsher channel table of this telescope (`table`), the checked
   limits in mK² (`limits`), and `coupling`, `curvature` and `margin`. Under fine decision v3,
   each channel gets its default word, the control's 1% word scaled to the channel, and a
   power-limit (BAO) word, or the reject word 1 where no word passes.
3. Test the bundle on later data with `test-bundle`, and qualify it with `qualify-bundle`
   (Step 3).
4. Compile the decisions with `compile-operating-policy`
   ([CHANNEL_OPERATING_POLICY.md](CHANNEL_OPERATING_POLICY.md)), then deploy the bundle
   (Step 4).

The power-limit test is `power_limit_word.limit_test`. It keeps the most frames for which

```
f * power_scale * (x + 2 s)^2 <= limit,    x = max(C_s e(u) - o - delta(u), 0)
```

where `e(u)` is the kept mean ratio excess `F / mu_0 - 1` with clustered error `s`, `o` is the
control's null and `delta(u)` its selection shift at the channel's word. The channel's
`curvature` is added to `o`. The upper estimate also adds two standard errors of `o`, `delta(u)`
and `curvature`, and the `margin`. The channel table supplies `power_scale`. The simple model
assigns temperature Tsys*x and spreads its squared amplitude uniformly over the supported
forecast modes. This normalization is a model convention, not an empirical sky calibration.
The word is in-sample, so it is tested on independent data before it masks.

`tools/threshold_rule.py` is an archive study tool. It runs the same sweep on a score cache
with the bias-ratio and finite-power criteria and applies no control corrections, so its words
differ from the power-limit test's and are not deployed.
