# Core calculation

| Quantity | Implementation |
|---|---|
| Fixed-point projection and fine spectrum | cuda/pilotproxy_detector.cu; fxfft.py |
| Frame powers and signed pilot excess | fine_reduction.py; product_contract.py |
| Exact rank score and Q16 keep boundary | fine_decision.py; detectors/narrowband_marker/scores.py |
| Defined archive population | characterization/eras.py and configured epochs |
| Authenticated score cache | characterization/score_cache.py |
| Retained proxy, clustered error, and maximum-retention threshold | characterization/threshold_rule.py |
| Receiver configuration and mask application | runtime_bundle.py; operating_policy.py; kotekan |

The response table comes from RFIsher's flat unit-additive-power model, schema
v3. A declared conversion places the measured proxy in its covariance units.
The score itself is not a power estimate. Detector validation, power-estimator
validation, and runtime mask correctness remain distinct checks.
