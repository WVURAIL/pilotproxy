# coding=utf-8
"""The false-discovery-rate default word of fine decision v3 (Benjamini and Hochberg 1995).

The control carries no DTV, and its scores give the null of the score. The null of channel c is the control scaled
by lambda_c = mu0_c g_c / (mu0_ref g_ref). For a threshold eta,
  E_0,c(eta)  the share of control frames with score above eta / lambda_c: the empirical share while at least
              N_EMPIRICAL control frames lie above, and beyond that a generalized Pareto tail fitted to the excesses
              over the empirical Q_U quantile (peaks over threshold);
  E_c(eta)    the share of the channel's frames with score above eta;
  FDR_c(eta)  pi0 E_0,c / E_c with pi0 = 1.
The threshold eta_c is the smallest score of the channel with FDR_c <= q, the step-up rule, and the word is
ceil(eta_c 2^16), the Q16 word of that frame. A channel with no such score gets the largest word, 2^64 - 1, which
masks nothing more: the detector still rejects a frame with d_rho = 0 (no decision) or a word above 2^64 - 1.

This is the rule of the dissertation (scripts/assemble_default_word.py and scripts/evt.py, Sec. 2.3), which takes
the frames recorded with the Sun below the horizon at DRAO, counted without capture weights. The GPD is fitted by
maximum likelihood through the profile likelihood in theta = xi / sigma (Grimshaw 1993).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

Q16 = 65536
# false discovery rate of the default word
FDR_LEVEL = 0.01
# share of null frames among the channel's frames, taken as one (conservative)
PI0 = 1.0
# the empirical null share holds while at least this many control frames lie above
N_EMPIRICAL = 20
# the GPD is fitted to the control frames above this empirical quantile
Q_U = 0.98
# the default word of a channel that masks nothing: the largest word of the bundle format (uint64). Frames with
# d_rho = 0 or a word above it are still rejected.
MASKS_NOTHING_WORD_Q16 = (1 << 64) - 1


class TooFewNullFrames(ValueError):
    """Too few null frames lie above the tail quantile for the GPD fit."""


def _profile(theta: float, y: np.ndarray) -> float:
    """Profile log-likelihood (per excess) at theta = xi/sigma, with xi(theta) = mean log(1 + theta*y)."""
    if abs(theta) < 1e-12:
        return -np.log(y.mean()) - 1.0
    xi = np.log1p(theta * y).mean()
    return -np.log(xi / theta) - xi - 1.0


def gpd_fit(y: np.ndarray) -> tuple[float, float]:
    """Return the maximum-likelihood (xi, sigma) of the GPD F(y) = 1 - (1 + xi y / sigma)^(-1/xi) for excesses y > 0."""
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as missing:
        raise ImportError("the false-discovery-rate default word needs scipy: pip install 'pilot-proxy[calibration]'") \
            from missing
    y = np.asarray(y, float)
    m, ymax = y.mean(), y.max()
    lo = -1 / ymax
    grid = np.sort(np.concatenate([lo * (1 - np.logspace(-8, 0, 60)[:-1]),   # (lo, 0)
                                   np.logspace(-4, 3, 120) / m]))
    grid = grid[grid > lo]
    with np.errstate(all="ignore"):
        xi = np.log1p(np.outer(grid, y)).mean(axis=1)
        ll = -np.log(xi / grid) - xi - 1.0
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    i = int(np.argmax(ll))
    a = grid[max(i - 1, 0)]
    b = grid[min(i + 1, grid.size - 1)]
    r = minimize_scalar(lambda t: -_profile(t, y), bounds=(a, b), method="bounded",
                        options={"xatol": 1e-10 * max(abs(a), abs(b), 1 / m)})
    theta = r.x if -r.fun >= ll[i] else grid[i]
    if abs(theta) < 1e-12:
        return 0.0, float(m)
    xi = np.log1p(theta * y).mean()
    return float(xi), float(xi / theta)


def gpd_survival(x, u: float, zeta: float, xi: float, sigma: float) -> np.ndarray:
    """Return the share above x > u, given P(X > u) = zeta and GPD(xi, sigma) excesses; zero past the end when xi < 0."""
    y = np.asarray(x, float) - u
    if abs(xi) < 1e-9:
        return zeta * np.exp(-y / sigma)
    return zeta * np.maximum(1 + xi * y / sigma, 0.0) ** (-1 / xi)


def pot(x: np.ndarray, q_u: float = Q_U) -> dict[str, float]:
    """Return the GPD fit of the excesses of x above its empirical q_u quantile: u, k, zeta = k / n, xi and sigma."""
    x = np.asarray(x, float)
    u = np.quantile(x, q_u)
    y = x[x > u] - u
    if y.size < N_EMPIRICAL:
        raise TooFewNullFrames(f"{y.size} null frames lie above the {q_u} quantile of {x.size}, and the tail fit "
                               f"needs {N_EMPIRICAL} strictly above it (about {int(np.ceil(N_EMPIRICAL / (1 - q_u)))} "
                               "null frames with a decision, more where scores tie): record more control frames, "
                               "or take the rule control_quantile")
    xi, sigma = gpd_fit(y)
    return dict(u=float(u), k=int(y.size), zeta=y.size / x.size, xi=xi, sigma=sigma)


class ControlNull:
    """E_0 of the control scores eta: the empirical share while N_EMPIRICAL frames lie above, the GPD tail beyond."""

    def __init__(self, eta: np.ndarray, *, n_empirical: int = N_EMPIRICAL, q_u: float = Q_U):
        eta = np.asarray(eta, float)
        self.sorted = np.sort(eta)
        self.n = eta.size
        self.n_empirical = int(n_empirical)
        # in the order given: the sums of the fit, and so its last digits, depend on the order of the excesses
        self.fit = pot(eta, q_u)
        # above this, fewer than n_empirical frames lie above
        self.tail_from = float(self.sorted[-self.n_empirical])

    def above(self, x) -> np.ndarray:
        return self.n - np.searchsorted(self.sorted, x, side="right")

    def __call__(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Return E_0 at x and whether each value is the empirical share (else the GPD tail)."""
        x = np.asarray(x, float)
        count = self.above(x)
        f = self.fit
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            tail = gpd_survival(x, f["u"], f["zeta"], f["xi"], f["sigma"])
        empirical = count >= self.n_empirical
        return np.where(empirical, count / self.n, tail), empirical


class Threshold(NamedTuple):
    """The default of one channel: its word (None when it masks nothing) and the estimates at it."""

    word: int | None
    channel_share: float
    control_share: float | None
    control_share_source: str | None
    fdr: float | None
    frames: int


def fdr_threshold(words: np.ndarray, scale: float, null: ControlNull, level: float = FDR_LEVEL,
                  pi0: float = PI0) -> Threshold:
    """Return the default of a channel from the Q16 keep words of its frames, at the control scale lambda_c.

    `words` are the frames with a decision. The word is the smallest word of the channel with
    pi0 E_0(eta / lambda_c) / E_c(eta) <= level, eta = word / 2^16, or None when no word qualifies.
    """
    words = np.sort(np.asarray(words, dtype=np.int64))
    if not words.size:
        return Threshold(None, 0.0, None, None, None, 0)
    grid = np.unique(words)
    eta = grid.astype(np.float64) / Q16
    channel_share = (words.size - np.searchsorted(words, grid, side="right")) / words.size
    control_share, empirical = null(eta / scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = np.where(channel_share > 0, pi0 * control_share / channel_share, np.inf)
    hit = np.flatnonzero(fdr <= level)
    if not hit.size:
        return Threshold(None, 0.0, None, None, None, int(words.size))
    i = int(hit[0])
    return Threshold(int(grid[i]), float(channel_share[i]), float(control_share[i]),
                     "empirical" if empirical[i] else "tail", float(fdr[i]), int(words.size))
