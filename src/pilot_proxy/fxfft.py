# coding=utf-8
"""fxfft256 v1: the frozen deterministic fixed-point FFT for the deployed
fine reduction.

This module is the *verification reference* for the fine detection path's
window-axis FFT (``docs/DESIGN_DECISIONS.md``, "The deployed decision is
the fine designated-set CFAR in the kernel"). The CUDA deployment must
reproduce these outputs bit-for-bit; ``tests/core/test_fxfft256.py`` holds
the golden-vector gate. The offline analyzer's float pipeline
(``fine_reduction.py``) is unchanged; this reference exists so the deployed
device FFT has a frozen, platform-independent definition that a port is
compared against, rather than a library FFT that cannot bit-match across
implementations.

Frozen specification (fxfft256 v1)
----------------------------------

* Transform: N = 256 forward DFT, ``sum_m x[m] e^{-2 pi i b m / 256}``
  (numpy sign convention), radix-2 decimation-in-time, bit-reversed load,
  natural-order output, in the exact butterfly order given below.
* Input: 128 complex int32 window sums (one stream's ``z[m]``,
  m = 0..127), zero-padded to 256. Contract: ``|re|, |im| <= 2**20``.
  Deployed row sums obey ``|.| <= K * 128 = 2**14`` (K = 128 samples,
  int4 x int4 product components bounded by 128), giving 6 bits of
  spare contract headroom.
* Twiddles: ``W[k] = (C[k], S[k])``, k = 0..127, representing
  ``e^{-2 pi i k / 256}`` in Q15:
  ``C[k] = nearest(32768 cos(2 pi k / 256))``,
  ``S[k] = nearest(-32768 sin(2 pi k / 256))``. The table below is the
  frozen source of truth (no runtime libm); the generator cross-check
  lives in the test suite. No value sits at a rounding tie. ``C[0] =
  32768`` and ``S[64] = -32768`` are stored in int32; multiplication by
  ``+-32768`` is exact under the rounding rule.
* Butterfly: ``t = round15(W[k] * b)``; ``(a, b) -> (a + t, a - t)`` in
  int32. Complex product components before rounding are exact int64.
* Rounding: ``round15(v) = floor((v + 16384) / 32768)`` per component,
  implemented as an arithmetic right shift by 15 after adding ``2**14``.
* Scaling: none. No-overflow proof: one butterfly maps the working
  infinity-norm ``M`` to at most ``(1 + sqrt(2) + 2**-15) M + 1``; eight
  stages from ``M <= 2**20`` stay below ``2**30.2 < 2**31 - 1``. The
  compile-time check in ``cuda/config.h`` and :func:`input_abs_max` use a
  looser form that adds one input norm per stage, which gives
  ``2**30.94`` and still closes. The
  implementation enforces the input contract and raises on any stage
  exceeding int32 rather than wrapping.

Downstream exactness: ``|X|**2`` and its sums over streams are computed
in exact (u)int64 (outputs are < 2**26 in magnitude for deployed inputs,
so squared sums over 2048 streams stay < 2**63). After the rounded FFT,
the deployed fine statistic is therefore deterministic integer arithmetic
end to end; the only departure from real-valued mathematics is the FFT
rounding itself, quantified in ``tools/fxfft_report.py``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

FXFFT256_SPEC_VERSION = "fxfft256_v1"

N = 256
N_IN = 128
SHIFT = 15
ROUND_CONST = 1 << 14
INPUT_ABS_MAX = 1 << 20
INT32_MAX = (1 << 31) - 1

# Frozen Q15 twiddle table: W[k] = (C[k], S[k]) ~ e^{-2 pi i k/256}.
# The source of truth is this literal table rather than runtime trigonometry.
TWIDDLE_Q15 = (
    (32768, 0), (32758, -804), (32729, -1608), (32679, -2411),
    (32610, -3212), (32522, -4011), (32413, -4808), (32286, -5602),
    (32138, -6393), (31972, -7180), (31786, -7962), (31581, -8740),
    (31357, -9512), (31114, -10279), (30853, -11039), (30572, -11793),
    (30274, -12540), (29957, -13279), (29622, -14010), (29269, -14733),
    (28899, -15447), (28511, -16151), (28106, -16846), (27684, -17531),
    (27246, -18205), (26791, -18868), (26320, -19520), (25833, -20160),
    (25330, -20788), (24812, -21403), (24279, -22006), (23732, -22595),
    (23170, -23170), (22595, -23732), (22006, -24279), (21403, -24812),
    (20788, -25330), (20160, -25833), (19520, -26320), (18868, -26791),
    (18205, -27246), (17531, -27684), (16846, -28106), (16151, -28511),
    (15447, -28899), (14733, -29269), (14010, -29622), (13279, -29957),
    (12540, -30274), (11793, -30572), (11039, -30853), (10279, -31114),
    (9512, -31357), (8740, -31581), (7962, -31786), (7180, -31972),
    (6393, -32138), (5602, -32286), (4808, -32413), (4011, -32522),
    (3212, -32610), (2411, -32679), (1608, -32729), (804, -32758),
    (0, -32768), (-804, -32758), (-1608, -32729), (-2411, -32679),
    (-3212, -32610), (-4011, -32522), (-4808, -32413), (-5602, -32286),
    (-6393, -32138), (-7180, -31972), (-7962, -31786), (-8740, -31581),
    (-9512, -31357), (-10279, -31114), (-11039, -30853), (-11793, -30572),
    (-12540, -30274), (-13279, -29957), (-14010, -29622), (-14733, -29269),
    (-15447, -28899), (-16151, -28511), (-16846, -28106), (-17531, -27684),
    (-18205, -27246), (-18868, -26791), (-19520, -26320), (-20160, -25833),
    (-20788, -25330), (-21403, -24812), (-22006, -24279), (-22595, -23732),
    (-23170, -23170), (-23732, -22595), (-24279, -22006), (-24812, -21403),
    (-25330, -20788), (-25833, -20160), (-26320, -19520), (-26791, -18868),
    (-27246, -18205), (-27684, -17531), (-28106, -16846), (-28511, -16151),
    (-28899, -15447), (-29269, -14733), (-29622, -14010), (-29957, -13279),
    (-30274, -12540), (-30572, -11793), (-30853, -11039), (-31114, -10279),
    (-31357, -9512), (-31581, -8740), (-31786, -7962), (-31972, -7180),
    (-32138, -6393), (-32286, -5602), (-32413, -4808), (-32522, -4011),
    (-32610, -3212), (-32679, -2411), (-32729, -1608), (-32758, -804),
)

_BITREV = tuple(int("{:08b}".format(i)[::-1], 2) for i in range(N))
_TW = np.asarray(TWIDDLE_Q15, dtype=np.int64)
_BITREV_IDX = np.asarray(_BITREV, dtype=np.int64)


def twiddle_table_sha256() -> str:
    """Hash of the frozen table literal (recorded in golden metadata)."""
    import hashlib

    return hashlib.sha256(str(list(TWIDDLE_Q15)).encode()).hexdigest()


def _validate_input(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind not in "iu":
        raise TypeError("fxfft256 input must be an integer array (re/im).")
    if a.ndim < 2 or a.shape[-1] != 2 or a.shape[-2] != N_IN:
        raise ValueError(
            f"fxfft256 input must have shape [..., {N_IN}, 2]; got {a.shape}."
        )
    wide = a.astype(np.int64, copy=False)
    if int(np.abs(wide).max(initial=0)) > INPUT_ABS_MAX:
        raise OverflowError(
            f"fxfft256 input exceeds the |.| <= 2**20 contract "
            f"(max {int(np.abs(wide).max())})."
        )
    return wide


def fxfft256(x: Any, *, return_stage_maxima: bool = False):
    """Frozen fixed-point 256-point FFT of 128-sample int rows.

    ``x``: integer array ``[..., 128, 2]`` (re, im). Returns int32
    ``[..., 256, 2]`` in natural bin order (numpy ``fft`` convention).
    With ``return_stage_maxima=True`` also returns the per-stage maximum
    absolute working value (headroom audit).
    """
    a = _validate_input(np.asarray(x))
    lead = a.shape[:-2]
    # zero-pad to 256 then load in bit-reversed order
    buf = np.zeros(lead + (N, 2), dtype=np.int64)
    buf[..., :N_IN, :] = a
    work = buf[..., _BITREV_IDX, :]

    maxima = []
    for s in range(1, 9):
        m = 1 << s
        half = m >> 1
        blocks = N >> s
        w = work.reshape(lead + (blocks, m, 2))
        ap = w[..., :half, :]
        bp = w[..., half:, :]
        k = (np.arange(half, dtype=np.int64)) << (8 - s)
        c = _TW[k, 0]
        sn = _TW[k, 1]
        br = bp[..., 0]
        bi = bp[..., 1]
        t_re = (br * c - bi * sn + ROUND_CONST) >> SHIFT
        t_im = (bi * c + br * sn + ROUND_CONST) >> SHIFT
        out = np.empty_like(w)
        out[..., :half, 0] = ap[..., 0] + t_re
        out[..., :half, 1] = ap[..., 1] + t_im
        out[..., half:, 0] = ap[..., 0] - t_re
        out[..., half:, 1] = ap[..., 1] - t_im
        work = out.reshape(lead + (N, 2))
        peak = int(np.abs(work).max(initial=0))
        maxima.append(peak)
        if peak > INT32_MAX:
            raise OverflowError(
                f"fxfft256 stage {s} exceeded int32 (peak {peak}); "
                "input violates the headroom analysis."
            )
    result = work.astype(np.int32)
    if return_stage_maxima:
        return result, maxima
    return result


# ---------------------------------------------------------------------------
# Transform family: one master twiddle table, every size by decimation
# ---------------------------------------------------------------------------
#
# A radix-2 DIT of length n needs W_n[k] = e^{-2 pi i k / n} for k < n/2 only.
# Because W_n[k] = W_M[k * M / n] whenever n divides M, and the Q15 rounding is
# applied to the same real value either way, the length-n table is an *exact*
# decimation of a length-M master: identical integers rather than an
# approximation.
# One master therefore serves the whole family, and the tie analysis and
# exact-rounding argument are established once on it and inherited by every
# member.
#
# MASTER_N is the largest transform the master serves. The deployed range needs
# L_F = 2 * N / K: with K = 128 on a CHIME-class raster and K = 64 on a
# CHORD-class one, frame sizes 2**13..2**16 give L_F in {256 .. 2048}. Raising
# MASTER_N is a one-constant change; the per-size no-overflow bound below is
# what actually limits usable sizes.

MASTER_N = 2048
MASTER_TWIDDLE_SHA256 = (
    "fec7f22309f4689a1a4a26258dc562487a02e33698a5e0341e2db1928f58d197"
)


def _generate_twiddle_half(n: int) -> tuple[tuple[int, int], ...]:
    """W_n[k] in Q15 for k < n/2, by exact nearest rounding.

    Used to build the master once at import. No entry sits at a rounding tie
    (the closest approach across the master is 5.8e-4, ~2.6e8 times the double
    epsilon at that scale), so every IEEE-754 libm produces the same integers;
    the hash check below turns that from an argument into an assertion.
    """
    import math

    out = []
    for k in range(n // 2):
        ang = 2.0 * math.pi * k / n
        out.append((round(32768.0 * math.cos(ang)), round(-32768.0 * math.sin(ang))))
    return tuple(out)


MASTER_TWIDDLE_Q15 = _generate_twiddle_half(MASTER_N)


def master_twiddle_sha256() -> str:
    """Hash of the master table (the family's single frozen artifact)."""
    import hashlib

    return hashlib.sha256(str(list(MASTER_TWIDDLE_Q15)).encode()).hexdigest()


def twiddle_table(n: int) -> tuple[tuple[int, int], ...]:
    """Q15 twiddles for a length-``n`` radix-2 DIT: ``n // 2`` entries.

    Returned by decimating the master with stride ``MASTER_N // n``; the
    values are bit-identical to generating length ``n`` directly.
    """
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"transform length must be a power of two >= 2; got {n}")
    if n > MASTER_N:
        raise ValueError(
            f"transform length {n} exceeds the master table (MASTER_N={MASTER_N}); "
            "raise MASTER_N and refresh MASTER_TWIDDLE_SHA256."
        )
    stride = MASTER_N // n
    return tuple(MASTER_TWIDDLE_Q15[k * stride] for k in range(n // 2))


# The frozen fxfft256 v1 literal must be exactly what the family produces at
# n = 256. This is the join between the frozen artifact and the general
# construction; if it ever fails, the fault is in the family rather than the literal table.
if master_twiddle_sha256() != MASTER_TWIDDLE_SHA256:
    raise RuntimeError(
        "master twiddle table does not match its frozen hash; the host libm "
        "disagrees with the recorded rounding, which would break bit-reproducibility"
    )
if twiddle_table(N) != TWIDDLE_Q15:
    raise RuntimeError(
        "master-table decimation does not reproduce the frozen fxfft256 twiddles"
    )

_BUTTERFLY_GROWTH = 1.0 + 1.41422 + (1.0 / 32768.0)


def input_abs_max(n: int) -> int:
    """Largest input infinity-norm with a proven no-overflow closure at ``n``.

    One butterfly maps the working norm ``M`` to at most
    ``(1 + sqrt(2) + 2**-15) M + 1``. This function applies the looser
    ``(1 + sqrt(2) + 2**-15) M + M_0`` (``M_0`` the input norm, so the
    ``+ 1`` is covered for any ``M_0 >= 1``) ``log2(n)`` times and requires
    the result below ``2**31``, which gives the bound, floored to a power
    of two. The bound *tightens* as ``n`` grows, so the eight-stage contract
    (``2**20``) is not reusable at nine stages, which is precisely the
    failure mode this function exists to prevent.
    """
    stages = n.bit_length() - 1
    norm = 1.0
    for _ in range(stages):
        norm = _BUTTERFLY_GROWTH * norm + 1.0
    return 1 << int(math.floor(math.log2((2 ** 31) / norm)))


def fxfft(x: Any, *, n_out: int = N, return_stage_maxima: bool = False):
    """Frozen fixed-point FFT generalized over the transform length.

    Same specification as :func:`fxfft256` (bit-reversed load, radix-2 DIT,
    Q15 twiddles from the master table, ``round15`` per butterfly, no
    scaling), with the length as a parameter. ``x`` is an integer array
    ``[..., n_out // 2, 2]``; the pad factor is fixed at two.

    At ``n_out = 256`` this is bit-identical to :func:`fxfft256`, which the
    test suite asserts.
    """
    if n_out < 2 or (n_out & (n_out - 1)) != 0:
        raise ValueError(f"n_out must be a power of two; got {n_out}")
    n_in = n_out // 2
    stages = n_out.bit_length() - 1
    limit = input_abs_max(n_out)

    a = np.asarray(x)
    if a.dtype.kind not in "iu":
        raise TypeError("fxfft input must be an integer array (re/im).")
    if a.ndim < 2 or a.shape[-1] != 2 or a.shape[-2] != n_in:
        raise ValueError(f"fxfft input must have shape [..., {n_in}, 2]; got {a.shape}")
    a = a.astype(np.int64, copy=False)
    peak_in = int(np.abs(a).max(initial=0))
    if peak_in > limit:
        raise OverflowError(
            f"fxfft input exceeds the |.| <= {limit} contract for n_out={n_out} "
            f"(max {peak_in}); the bound tightens with transform length."
        )

    width = stages
    bitrev = np.asarray(
        [int(format(i, "0%db" % width)[::-1], 2) for i in range(n_out)], dtype=np.int64
    )
    tw = np.asarray(twiddle_table(n_out), dtype=np.int64)

    lead = a.shape[:-2]
    buf = np.zeros(lead + (n_out, 2), dtype=np.int64)
    buf[..., :n_in, :] = a
    work = buf[..., bitrev, :]

    maxima = []
    for s in range(1, stages + 1):
        m = 1 << s
        half = m >> 1
        blocks = n_out >> s
        w = work.reshape(lead + (blocks, m, 2))
        ap = w[..., :half, :]
        bp = w[..., half:, :]
        k = (np.arange(half, dtype=np.int64)) << (stages - s)
        c = tw[k, 0]
        sn = tw[k, 1]
        br = bp[..., 0]
        bi = bp[..., 1]
        t_re = (br * c - bi * sn + ROUND_CONST) >> SHIFT
        t_im = (bi * c + br * sn + ROUND_CONST) >> SHIFT
        out = np.empty_like(w)
        out[..., :half, 0] = ap[..., 0] + t_re
        out[..., :half, 1] = ap[..., 1] + t_im
        out[..., half:, 0] = ap[..., 0] - t_re
        out[..., half:, 1] = ap[..., 1] - t_im
        work = out.reshape(lead + (n_out, 2))
        peak = int(np.abs(work).max(initial=0))
        maxima.append(peak)
        if peak > INT32_MAX:
            raise OverflowError(
                f"fxfft(n_out={n_out}) stage {s} exceeded int32 (peak {peak})."
            )
    result = work.astype(np.int32)
    return (result, maxima) if return_stage_maxima else result


def fxfft256_scalar(x_in) -> list[tuple[int, int]]:
    """Pure-scalar reference (mirrors ``cuda/fxfft256_ref.c`` structure).

    ``x_in``: sequence of 128 ``(re, im)`` integer pairs. Returns a list of
    256 ``(re, im)`` pairs. Bit-identical to :func:`fxfft256`.
    """
    xs = [(int(r), int(i)) for r, i in x_in]
    if len(xs) != N_IN:
        raise ValueError("fxfft256_scalar expects 128 (re, im) pairs.")
    for r, i in xs:
        if abs(r) > INPUT_ABS_MAX or abs(i) > INPUT_ABS_MAX:
            raise OverflowError("input exceeds the |.| <= 2**20 contract.")
    padded = xs + [(0, 0)] * (N - N_IN)
    x = [padded[_BITREV[j]] for j in range(N)]
    for s in range(1, 9):
        m = 1 << s
        half = m >> 1
        for j0 in range(0, N, m):
            for t in range(half):
                c, sn = TWIDDLE_Q15[t << (8 - s)]
                ar, ai = x[j0 + t]
                br, bi = x[j0 + t + half]
                tr = (br * c - bi * sn + ROUND_CONST) >> SHIFT
                ti = (bi * c + br * sn + ROUND_CONST) >> SHIFT
                x[j0 + t] = (ar + tr, ai + ti)
                x[j0 + t + half] = (ar - tr, ai - ti)
    return x


def fine_power_fx(
    matched_filter_row_projections: Any,
    *,
    num_streams: int,
    windows_per_stream: int = N_IN,
    num_weight_terms: int = 3,
) -> np.ndarray:
    """Exact-integer fine power spectra via the frozen FFT.

    ``matched_filter_row_projections``: integer array ``[terms, streams * windows, 2]`` (the
    kernel's stream-major row-sum layout, as in ``fine_reduction``).
    Returns uint64 ``S[terms, 256]`` = sum over streams of ``|X|**2`` ---
    exact integers, the deployed fine-statistic numerators/denominators.
    """
    if int(windows_per_stream) != N_IN:
        raise ValueError("fxfft256 is frozen at 128 windows per stream.")
    arr = np.asarray(matched_filter_row_projections)
    terms = int(num_weight_terms)
    streams = int(num_streams)
    if arr.ndim != 3 or arr.shape[0] != terms or arr.shape[-1] != 2:
        raise ValueError("matched_filter_row_projections must have shape [terms, rows, 2].")
    if arr.shape[1] != streams * N_IN:
        raise ValueError("rows must equal num_streams * 128.")
    z = arr.reshape(terms, streams, N_IN, 2)
    X = fxfft256(z).astype(np.int64)
    mag = X[..., 0] * X[..., 0] + X[..., 1] * X[..., 1]
    return mag.sum(axis=1, dtype=np.uint64)


def fine_power_ratio_fx(power: np.ndarray) -> np.ndarray:
    """``F2[b] = 2 S_t / (S_l + S_u)`` from exact fx powers (float64 out)."""
    p = np.asarray(power, dtype=np.float64)
    den = p[1] + p[2]
    return np.where(den > 0, 2.0 * p[0] / np.where(den > 0, den, 1.0), 0.0)
