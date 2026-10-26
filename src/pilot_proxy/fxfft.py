# coding=utf-8
"""fxfft256 v1, the frozen deterministic fixed-point FFT of the deployed
fine reduction.

This module is the verification reference for the window-axis FFT of the
fine detection path (``docs/DESIGN_DECISIONS.md``, "The deployed decision
is the fine designated-set CFAR in the kernel"). The CUDA deployment must
reproduce these outputs bit for bit, and ``tests/core/test_fxfft256.py``
holds the golden-vector gate. In the offline analyzer, the float pipeline
(``fine_reduction.py``) is a different code path. A library FFT cannot
bit-match across implementations, so this reference gives the deployed
device FFT a frozen, platform-independent definition to compare a port
against.

Frozen specification (fxfft256 v1)
----------------------------------

* Transform: N = 256 forward DFT, ``sum_m x[m] e^{-2 pi i b m / 256}``
  (numpy sign convention), radix-2 decimation-in-time, bit-reversed load,
  natural-order output, in the exact butterfly order given below.
* Input: 128 complex int32 window sums (``z[m]`` of one stream,
  m = 0..127), zero-padded to 256. Its contract is
  ``|re|, |im| <= 2**20``. Deployed row sums obey
  ``|.| <= K * 128 = 2**14`` (K = 128 samples, int4 x int4 product
  components bounded by 128), which leaves 6 bits of spare contract
  headroom.
* Twiddles: ``W[k] = (C[k], S[k])``, k = 0..127, representing
  ``e^{-2 pi i k / 256}`` in Q15, with
  ``C[k] = nearest(32768 cos(2 pi k / 256))`` and
  ``S[k] = nearest(-32768 sin(2 pi k / 256))``. The table below is the
  frozen source of truth, and no runtime libm is used. A generator
  cross-check is in the test suite. No value sits at a rounding tie.
  ``C[0] = 32768`` and ``S[64] = -32768`` are stored in int32, and
  multiplication by ``+-32768`` is exact under the rounding rule.
* Butterfly: ``t = round15(W[k] * b)`` and ``(a, b) -> (a + t, a - t)``
  in int32. Before rounding, the complex product components are exact
  int64.
* Rounding: ``round15(v) = floor((v + 16384) / 32768)`` per component,
  implemented as an arithmetic right shift by 15 after adding ``2**14``.
* Scaling: none. For the no-overflow proof, one butterfly maps the
  working infinity-norm ``M`` to at most ``(1 + sqrt(2) + 2**-15) M + 1``,
  and eight stages from ``M <= 2**20`` stay below ``2**30.2 < 2**31 - 1``.
  The compile-time check in ``cuda/config.h`` and :func:`input_abs_max`
  use a looser form that adds one input norm per stage. This form gives
  ``2**30.94`` and still closes. In the implementation, the input
  contract is enforced and any stage that exceeds int32 raises an error,
  so values never wrap.

The squared magnitudes ``|X|**2`` and their sums over streams are computed
in exact (u)int64. For deployed inputs the outputs are below 2**26 in
magnitude, so squared sums over 2048 streams stay below 2**63. After the
rounded FFT, the deployed fine statistic is therefore deterministic
integer arithmetic end to end. Only the FFT rounding departs from
real-valued mathematics, and it is quantified in ``tools/fxfft_report.py``.
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

# Frozen Q15 twiddle table, W[k] = (C[k], S[k]) ~ e^{-2 pi i k/256}. This
# literal table is the source of truth, and no runtime trigonometry is used.
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
    """Return the SHA256 of the frozen table literal (recorded in the golden metadata)."""
    import hashlib

    return hashlib.sha256(str(list(TWIDDLE_Q15)).encode()).hexdigest()


def _max_component_magnitude(a: np.ndarray) -> int:
    # The extremes are read in the original integer dtype and compared as
    # Python ints, since a cast of uint64 can wrap and a fixed-width abs
    # cannot represent the magnitude of INT64_MIN.
    return max(-int(a.min(initial=0)), int(a.max(initial=0)))


def _validate_input(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind not in "iu":
        raise TypeError("fxfft256 input must be an integer array (re/im).")
    if a.ndim < 2 or a.shape[-1] != 2 or a.shape[-2] != N_IN:
        raise ValueError(
            f"fxfft256 input must have shape [..., {N_IN}, 2]; got {a.shape}."
        )
    peak_in = _max_component_magnitude(a)
    if peak_in > INPUT_ABS_MAX:
        raise OverflowError(
            f"fxfft256 input exceeds the |.| <= 2**20 contract "
            f"(max {peak_in})."
        )
    return a.astype(np.int64, copy=False)


def fxfft256(x: Any, *, return_stage_maxima: bool = False):
    """Return the frozen fixed-point 256-point FFT of 128-sample integer rows.

    ``x`` is an integer array ``[..., 128, 2]`` (re, im). Returns int32
    ``[..., 256, 2]`` in natural bin order (numpy ``fft`` convention).
    With ``return_stage_maxima=True``, the per-stage maximum absolute
    working value is also returned for the headroom audit.
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


# Transform family from one master twiddle table by decimation
#
# A radix-2 DIT of length n needs W_n[k] = e^{-2 pi i k / n} only for k < n/2.
# Since W_n[k] = W_M[k * M / n] whenever n divides M, and the Q15 rounding is
# applied to the same real value in both cases, the length-n table is an exact
# decimation of a length-M master and holds identical integers. One master
# therefore serves the whole family. The tie analysis and the exact-rounding
# argument are made once on the master and hold for every member.
#
# MASTER_N is the largest transform the master serves. The deployed range needs
# L_F = 2 * N / K. With K = 128 on a CHIME-class raster and K = 64 on a
# CHORD-class raster, frame sizes 2**13 to 2**16 give L_F from 256 to 2048.
# Raising MASTER_N changes one constant, and the usable sizes are limited by
# the per-size no-overflow bound below.

MASTER_N = 2048
MASTER_TWIDDLE_SHA256 = (
    "fec7f22309f4689a1a4a26258dc562487a02e33698a5e0341e2db1928f58d197"
)


def _generate_twiddle_half(n: int) -> tuple[tuple[int, int], ...]:
    """Return W_n[k] in Q15 for k < n/2, by exact nearest rounding.

    The master is built with this function once at import. No entry sits at
    a rounding tie. The closest approach over the master is 5.8e-4, about
    2.6e8 times the double epsilon at that scale, so every IEEE-754 libm
    produces the same integers. The hash check below asserts this at import.
    """
    import math

    out = []
    for k in range(n // 2):
        ang = 2.0 * math.pi * k / n
        out.append((round(32768.0 * math.cos(ang)), round(-32768.0 * math.sin(ang))))
    return tuple(out)


MASTER_TWIDDLE_Q15 = _generate_twiddle_half(MASTER_N)


def master_twiddle_sha256() -> str:
    """Return the SHA256 of the master table, the single frozen artifact of the family."""
    import hashlib

    return hashlib.sha256(str(list(MASTER_TWIDDLE_Q15)).encode()).hexdigest()


def twiddle_table(n: int) -> tuple[tuple[int, int], ...]:
    """Return the ``n // 2`` Q15 twiddles of a length-``n`` radix-2 DIT.

    The table is decimated from the master with stride ``MASTER_N // n``,
    and its values are bit-identical to a table generated at length ``n``
    directly.
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


# The frozen fxfft256 v1 literal must equal the table the family produces at
# n = 256. This check joins the frozen artifact to the general construction.
# The literal table is the source of truth, so a failure here is a fault in
# the family.
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
    """Return a power-of-two input bound that prevents int32 overflow at length n.

    A butterfly takes the working norm M to at most
    (1 + sqrt(2) + 2**-15) M + 1. For an input norm M_0 >= 1, the looser
    bound with +M_0 in place of +1 is used. This bound is applied log2(n)
    times, and the final value must stay below 2**31.

    Longer transforms need smaller input bounds, so the eight-stage limit
    2**20 does not hold for nine stages.
    """
    stages = n.bit_length() - 1
    norm = 1.0
    for _ in range(stages):
        norm = _BUTTERFLY_GROWTH * norm + 1.0
    return 1 << int(math.floor(math.log2((2 ** 31) / norm)))


def fxfft(x: Any, *, n_out: int = N, return_stage_maxima: bool = False):
    """Return the frozen fixed-point FFT for a general transform length.

    The specification is that of :func:`fxfft256` (bit-reversed load,
    radix-2 DIT, Q15 twiddles from the master table, ``round15`` per
    butterfly, no scaling), with the length as a parameter. ``x`` is an
    integer array ``[..., n_out // 2, 2]``, so the pad factor is fixed at
    two.

    At ``n_out = 256`` the result is bit-identical to :func:`fxfft256`, and
    the test suite asserts this.
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
    peak_in = _max_component_magnitude(a)
    if peak_in > limit:
        raise OverflowError(
            f"fxfft input exceeds the |.| <= {limit} contract for n_out={n_out} "
            f"(max {peak_in}); the bound tightens with transform length."
        )
    a = a.astype(np.int64, copy=False)

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
    """Pure-scalar reference with the structure of ``cuda/fxfft256_ref.c``.

    ``x_in`` is a sequence of 128 ``(re, im)`` integer pairs. Returns a list
    of 256 ``(re, im)`` pairs, bit-identical to :func:`fxfft256`.
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
    """Return exact-integer fine power spectra from the frozen FFT.

    ``matched_filter_row_projections`` is an integer array
    ``[terms, streams * windows, 2]`` in the stream-major row-sum layout of
    the kernel, as in ``fine_reduction``. Returns uint64 ``S[terms, 256]``,
    the sum over streams of ``|X|**2``. These exact integers are the
    numerators and denominators of the deployed fine statistic.
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
