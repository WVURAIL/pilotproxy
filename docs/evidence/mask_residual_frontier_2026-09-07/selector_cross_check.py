"""Cross-check pilot_proxy.testbench.mask_frontier against the deployed RFIsher rules."""
import sys, json, math
sys.path.insert(0, "/home/djg/rail/pilot-proxy/src")
import numpy as np
from rfisher.residual_scores.bundle import required_multipliers_for_frame
from rfisher.preparation import candidate_multiplier_q16_values
from rfisher.thresholds import build_q16_residual_score_histogram, optimize_threshold
from rfisher_results.archive.selection import kept_at as rf_kept_at
from pilot_proxy.fine_reduction import independent_bin_mask
from pilot_proxy.testbench.mask_frontier import (
    required_multipliers_by_rank, candidate_multipliers_q16, kept_at,
    residual_score_histogram, frontier_points, ALWAYS_MASKED_Q16)
from pilot_proxy.testbench.sensitivity_study import designated_bins

rng = np.random.default_rng(20260907)
designated = designated_bins(255, 2)
bulk = independent_bin_mask(256, pad_factor=2, designated_bins=designated, guard_fine_bins=1)
n_bulk = int(bulk.sum())

mismatch_req = 0
frames = 300
required_by_rho = {}
for trial in range(frames):
    powers = rng.integers(1, 10**7, size=(3, 256)).astype(np.uint64)
    mine = required_multipliers_by_rank(powers, designated=designated, bulk_mask=bulk)
    theirs = required_multipliers_for_frame(powers, anchor_bin=255,
        designated_half_width=2, bulk_mask=np.ascontiguousarray(bulk))
    if tuple(mine) != tuple(theirs):
        mismatch_req += 1
    for rho in range(1, n_bulk + 1):
        required_by_rho.setdefault(rho, []).append(int(mine[rho - 1]))
print("required-multiplier frames compared:", frames, "bulk ranks:", n_bulk, "mismatches:", mismatch_req)

# candidate grid
for rho in (1, n_bulk // 2, n_bulk):
    a = candidate_multipliers_q16(required_by_rho[rho])
    b = candidate_multiplier_q16_values(required_by_rho[rho])
    assert a == b, rho
print("candidate staircase identical at rho in {1, %d, %d}" % (n_bulk // 2, n_bulk))

# keep rule
req = np.asarray(required_by_rho[n_bulk // 2] + [ALWAYS_MASKED_Q16], dtype=object)
for eta in (1, 65536, 200000, (1 << 64) - 1):
    mine = kept_at(req, eta)
    theirs = rf_kept_at(np.asarray([int(v) for v in req], dtype=object), eta)
    assert list(mine) == list(theirs), eta
print("keep rule identical at four multipliers")

# histogram + frontier walk against optimize_threshold
claim = rng.uniform(1e-6, 1e-2, frames)
rho = n_bulk // 2
cand = candidate_multipliers_q16(required_by_rho[rho])
h_mine = residual_score_histogram(required_by_rho[rho], claim, np.zeros(frames),
                                  candidates=cand, rho=rho, bulk_size=n_bulk)
h_theirs = build_q16_residual_score_histogram(required_by_rho[rho], claim, cand, bulk_size=n_bulk)
assert h_mine.counts == h_theirs.counts
assert np.allclose(h_mine.systematic_sums, h_theirs.systematic_residual_sums, rtol=0, atol=0)
opt = optimize_threshold({rho: h_theirs}, 1.0)
mine_pts = {p["eta_q16"]: p for p in frontier_points(h_mine)}
theirs_pts = {p.multiplier_q16: p for p in opt.points}
assert set(mine_pts) == set(theirs_pts), (len(mine_pts), len(theirs_pts))
worst_m = max(abs(mine_pts[k]["masked_fraction"] - theirs_pts[k].masked_fraction) for k in mine_pts)
worst_r = max(abs(mine_pts[k]["r_sys"] - theirs_pts[k].systematic_residual) for k in mine_pts)
print(f"frontier points compared: {len(mine_pts)}  max |dm| = {worst_m:.3e}  max |dr_sys| = {worst_r:.3e}")
print("CROSS-CHECK OK" if (mismatch_req == 0 and worst_m == 0.0 and worst_r == 0.0) else "CROSS-CHECK FAILED")
