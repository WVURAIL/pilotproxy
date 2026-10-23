# First run: from a clean checkout to a paper result

This walkthrough uses saved inputs. It needs Git, Python 3.10 or newer with
`venv`, and access to the repositories. It does not need CUDA, telescope data,
FIR, or CANFAR. Commands below assume a Linux shell.

## 1. Install and check the detector package

Start in a new working directory. Authenticate to GitHub first if a repository
is private; that is separate from telescope archive credentials.

```sh
git clone https://github.com/WVURAIL/pilotproxy.git
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e "pilotproxy[test]"
cd pilotproxy
pilot-proxy --help
python -m pytest tests/test_cli.py -q
pilot-proxy check-profile \
    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
pilot-proxy check-layout \
    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
```

The tests and both validation commands should pass. This checks installation
and receiver geometry, not GPU arithmetic. For CUDA builds and native tests,
continue with [installation and usage](USAGE.md).

## 2. Read and rebuild one forecast channel

PilotProxy measures the pilot and assigns a threshold. **RFIsher** supplies the
channel's response to residual DTV power. **RadioFisher** is the underlying
clean cosmological forecast. Install RFIsher alongside this checkout:

```sh
cd ..
git clone https://github.com/WVURAIL/RFIsher.git
python -m pip install -e "RFIsher[test]"
cd RFIsher
python - <<'PY'
import rfisher
row = next(r for r in rfisher.read_table()["channels"] if r["channel"] == 19)
print(f"Channel 19: response = {row['response_dv']:.0f}, a_tol = {row['a_tol']:.3g}")
PY
rfisher table --channels 19 --workers 1 --out out/first-run
rfisher figures --table out/first-run/channel_table.json --out out/first-run/figures
python -m pytest tests/test_table.py -q
```

The shipped row prints approximately `response = 8344, a_tol = 0.00012`.
The rebuild writes JSON and CSV files; the figure command explicitly reads
that new table. The tests compare channel 19 with the shipped result using
documented numerical tolerances. The forecast needs more time than reading
the table, but it runs one channel on one worker.

RFIsher installs its recorded RadioFisher revision automatically. Do not
replace that dependency with RadioFisher's latest source in this environment.
To develop RadioFisher itself, use its README in a separate environment.

## 3. Reproduce the paper's threshold decisions

Clone the MNRAS manuscript beside the two code repositories, then run the
read-only check:

```sh
cd ..
git clone https://github.com/WVURAIL/dtv-mnras.git
cd pilotproxy
python tools/check_paper_thresholds.py --paper-root ../dtv-mnras
```

Success is JSON with `"status": "pass"` and `"channels_checked": 24`. It checks
the paper's saved sweeps, responses, budgets, threshold words, and current
calendar-year comparison. It also checks the historical calculation using its
recorded time convention. No data refresh or TeX installation is needed.

These are conditional threshold calculations. Reproducing them does not
measure the assumed temporal gain `G`, qualify the transfer of `gamma` to a
new data population, or complete a live telescope run.

## 4. Trace a result and choose the next workflow

Follow [the paper/code map](PAPER_CODE_MAP.md) from each defined quantity to
its implementation and tests. For the threshold calculation, start with
`strength`, `sweep`, and `decide_guarded` in
[`characterization/threshold_rule.py`](../src/pilot_proxy/characterization/threshold_rule.py).
The [terminology reference](TERMINOLOGY.md) explains why the production score
is a marker-to-reference ratio rather than generally an F statistic.

- To process new observations, follow the [pipeline guide](START_HERE.md), then
  the [local](LOCAL_PROCESSING.md) or [CANFAR](CANFAR_RUNBOOK.md) runbook.
- To build the manuscript, use the README in `dtv-mnras`. Its saved data are
  separate from the original archive products.
- To work on receiver integration, use the [operator handoff](OPERATOR_HANDOFF.md).
  Match the vendored core and native API versions before building kotekan.

Keep generated results under `out/` or another output directory. Dated evidence
and frozen paper inputs identify the original measurements; leave those in
place when testing newer code.
