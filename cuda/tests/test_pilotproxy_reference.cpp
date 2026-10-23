#include "pilotproxy_detector.h"
#include "pilotproxy_reference.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

constexpr int INT4_COMPONENT_MASK = 0xF;
constexpr int INT4_REAL_SHIFT_BITS = 4;
constexpr int INT4_ZERO_CODE = 0x0;
constexpr int INT4_MAX_POSITIVE_CODE = 0x7;
constexpr int INT4_MIN_NEGATIVE_CODE = 0x8;
constexpr int INT4_NEGATIVE_ONE_CODE = 0xF;
constexpr int INT4_ZERO_VALUE = 0;
constexpr int INT4_MAX_POSITIVE_VALUE = 7;
constexpr int INT4_MIN_NEGATIVE_VALUE = -8;
constexpr int INT4_NEGATIVE_ONE_VALUE = -1;
constexpr int UNINITIALIZED_SENTINEL = 999;
constexpr int SINGLE_STREAM = 1;
constexpr int TWO_STREAMS = 2;
constexpr int WINDOWS = PILOTPROXY_FINE_WINDOWS_PER_STREAM;
constexpr int TERMS = PILOTPROXY_NUM_WEIGHT_TERMS;
constexpr int BINS = PILOTPROXY_FINE_NUM_BINS;

constexpr unsigned long long ZERO_POWER = 0ULL;
constexpr unsigned long long UNIT_POWER = 1ULL;
// The frozen transform's Parseval departure on the golden vectors is bounded
// at 5e-3 (tests/core/test_fxfft256.py); random rows at the amplitudes
// below sit far inside it. This documents the relation P ~ 256 sum|z|^2
// without asserting an equality the fixed-point transform does not provide.
constexpr double PARSEVAL_RELATIVE_TOLERANCE = 5.0e-3;

InputType pack_i4(int real, int imag)
{
    return static_cast<InputType>(
        ((real & INT4_COMPONENT_MASK) << INT4_REAL_SHIFT_BITS)
        | (imag & INT4_COMPONENT_MASK));
}

void assert_unpack(InputType packed, int expected_real, int expected_imag)
{
    int real = 0;
    int imag = 0;
    pilotproxy_ref_unpack_complex_i4(packed, &real, &imag);
    assert(real == expected_real);
    assert(imag == expected_imag);
}

void assert_mul(
    int xr,
    int xi,
    int wr,
    int wi,
    int expected_real,
    int expected_imag)
{
    int real = UNINITIALIZED_SENTINEL;
    int imag = UNINITIALIZED_SENTINEL;
    pilotproxy_ref_complex_mul_conj(xr, xi, wr, wi, &real, &imag);
    assert(real == expected_real);
    assert(imag == expected_imag);
}

// A unit impulse in window 0 of one stream transforms to a flat spectrum of
// exactly one per bin: the impulse only ever enters a butterfly as the top
// leg and every bottom leg is an exact zero, so no rounding occurs. The fine
// reference must therefore give S[f] = 1 for all 256 bins and the frame
// power the transform length; the untouched terms stay zero.
void assert_impulse_fine_powers(int component)
{
    std::vector<int> row_sums(static_cast<std::size_t>(TERMS) * WINDOWS * 2, 0);
    row_sums[static_cast<std::size_t>(component)] = 1;  // window 0 of the target term

    unsigned long long fine[TERMS][BINS];
    pilotproxy_ref_fine_powers_u64(row_sums.data(), SINGLE_STREAM, fine);
    for (int f = 0; f < BINS; ++f) {
        assert(fine[PILOTPROXY_TARGET_WEIGHT_INDEX][f] == UNIT_POWER);
        assert(fine[PILOTPROXY_LOWER_REFERENCE_WEIGHT_INDEX][f] == ZERO_POWER);
        assert(fine[PILOTPROXY_UPPER_REFERENCE_WEIGHT_INDEX][f] == ZERO_POWER);
    }

    unsigned long long powers[TERMS];
    pilotproxy_ref_powers_from_fine_u64(fine, powers);
    assert(powers[PILOTPROXY_TARGET_WEIGHT_INDEX] == static_cast<unsigned long long>(BINS));
    assert(powers[PILOTPROXY_LOWER_REFERENCE_WEIGHT_INDEX] == ZERO_POWER);
    assert(powers[PILOTPROXY_UPPER_REFERENCE_WEIGHT_INDEX] == ZERO_POWER);

    // Two streams carrying the same impulse: the feed sum doubles exactly.
    std::vector<int> two(static_cast<std::size_t>(TERMS) * TWO_STREAMS * WINDOWS * 2, 0);
    two[static_cast<std::size_t>(component)] = 1;
    two[static_cast<std::size_t>(WINDOWS) * 2 + component] = 1;
    pilotproxy_ref_fine_powers_u64(two.data(), TWO_STREAMS, fine);
    pilotproxy_ref_powers_from_fine_u64(fine, powers);
    for (int f = 0; f < BINS; ++f) {
        assert(fine[PILOTPROXY_TARGET_WEIGHT_INDEX][f] == 2ULL * UNIT_POWER);
    }
    assert(powers[PILOTPROXY_TARGET_WEIGHT_INDEX] == 2ULL * BINS);
}

void assert_null_arguments_are_harmless()
{
    unsigned long long fine[TERMS][BINS];
    for (int n = 0; n < TERMS; ++n) {
        for (int f = 0; f < BINS; ++f) {
            fine[n][f] = 7ULL;
        }
    }
    pilotproxy_ref_fine_powers_u64(nullptr, SINGLE_STREAM, fine);
    for (int n = 0; n < TERMS; ++n) {
        for (int f = 0; f < BINS; ++f) {
            assert(fine[n][f] == ZERO_POWER);
        }
    }
    pilotproxy_ref_fine_powers_u64(nullptr, SINGLE_STREAM, nullptr);
    pilotproxy_ref_powers_from_fine_u64(fine, nullptr);

    unsigned long long powers[TERMS] = {5ULL, 5ULL, 5ULL};
    pilotproxy_ref_powers_from_fine_u64(nullptr, powers);
    for (int n = 0; n < TERMS; ++n) {
        assert(powers[n] == ZERO_POWER);
    }
}

// The frame power is the exact all-bin sum: a hand-built spectrum sums to
// the closed form, including the uint64 range above 2^32 per bin.
void assert_frame_power_is_the_all_bin_sum()
{
    unsigned long long fine[TERMS][BINS];
    for (int n = 0; n < TERMS; ++n) {
        for (int f = 0; f < BINS; ++f) {
            fine[n][f] = (1ULL << 40) * static_cast<unsigned long long>(n + 1)
                + static_cast<unsigned long long>(f);
        }
    }
    unsigned long long powers[TERMS];
    pilotproxy_ref_powers_from_fine_u64(fine, powers);
    const unsigned long long bin_sum =
        static_cast<unsigned long long>(BINS) * (BINS - 1) / 2ULL;
    for (int n = 0; n < TERMS; ++n) {
        assert(powers[n] == (1ULL << 40) * BINS * (n + 1) + bin_sum);
    }
}

}  // namespace

static unsigned int lcg_next(unsigned int* state)
{
    *state = (*state) * 1664525u + 1013904223u;
    return (*state) >> 16;
}

static void run_row_sums_reference_checks()
{
    const int rows = 37;
    unsigned int state = 0xC0FFEEu;

    std::vector<InputType> x(
        static_cast<std::size_t>(rows) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        static_cast<std::size_t>(PILOTPROXY_NUM_WEIGHT_TERMS)
        * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int re = static_cast<int>(lcg_next(&state) % 15) - 7;
        const int im = static_cast<int>(lcg_next(&state) % 15) - 7;
        x[i] = pack_i4(re, im);
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        const int re = static_cast<int>(lcg_next(&state) % 15) - 7;
        const int im = static_cast<int>(lcg_next(&state) % 15) - 7;
        w[i] = pack_i4(re, im);
    }

    std::vector<int> row_sums(
        static_cast<std::size_t>(PILOTPROXY_NUM_WEIGHT_TERMS) * rows * 2, 0);
    pilotproxy_ref_row_sums_i32(x.data(), w.data(), rows, row_sums.data());

    // Exact equality against an independently ordered brute force
    // (term-outer, tap-outer accumulation): integer addition is associative,
    // so any ordering must agree bit-for-bit.
    for (int n = 0; n < PILOTPROXY_NUM_WEIGHT_TERMS; ++n) {
        for (int m = 0; m < rows; ++m) {
            long long br = 0;
            long long bi = 0;
            for (int k = 0; k < PILOTPROXY_DETECTOR_WINDOW_SAMPLES; ++k) {
                int xr;
                int xi;
                int wr;
                int wi;
                int yr;
                int yi;
                pilotproxy_ref_unpack_complex_i4(
                    x[static_cast<std::size_t>(m)
                          * PILOTPROXY_DETECTOR_WINDOW_SAMPLES
                      + k],
                    &xr,
                    &xi);
                pilotproxy_ref_unpack_complex_i4(
                    w[static_cast<std::size_t>(n)
                          * PILOTPROXY_DETECTOR_WINDOW_SAMPLES
                      + k],
                    &wr,
                    &wi);
                pilotproxy_ref_complex_mul_conj(xr, xi, wr, wi, &yr, &yi);
                br += yr;
                bi += yi;
            }
            const std::size_t idx =
                (static_cast<std::size_t>(n) * rows + m) * 2;
            assert(static_cast<long long>(row_sums[idx + 0]) == br);
            assert(static_cast<long long>(row_sums[idx + 1]) == bi);
        }
    }
}

// Random rows through the whole reference chain: row sums -> fine spectra ->
// frame powers. Checks (a) the frame power is the exact all-bin sum of the
// fine spectrum, (b) the feed sum is additive over streams (the stream
// indexing of the term-major buffer is right), and (c) the Parseval relation
// P ~ L_F * sum|z|^2 holds to the frozen transform's rounding.
static void run_fine_reference_checks()
{
    const int rows = TWO_STREAMS * WINDOWS;
    unsigned int state = 0xF1DE5Au;

    std::vector<InputType> x(
        static_cast<std::size_t>(rows) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        static_cast<std::size_t>(TERMS) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int re = static_cast<int>(lcg_next(&state) % 15) - 7;
        const int im = static_cast<int>(lcg_next(&state) % 15) - 7;
        x[i] = pack_i4(re, im);
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        const int re = static_cast<int>(lcg_next(&state) % 15) - 7;
        const int im = static_cast<int>(lcg_next(&state) % 15) - 7;
        w[i] = pack_i4(re, im);
    }

    std::vector<int> row_sums(static_cast<std::size_t>(TERMS) * rows * 2, 0);
    pilotproxy_ref_row_sums_i32(x.data(), w.data(), rows, row_sums.data());

    unsigned long long fine[TERMS][BINS];
    unsigned long long powers[TERMS];
    pilotproxy_ref_fine_powers_u64(row_sums.data(), TWO_STREAMS, fine);
    pilotproxy_ref_powers_from_fine_u64(fine, powers);

    // (a) definition: P_n == sum_f S_n[f], summed independently here.
    for (int n = 0; n < TERMS; ++n) {
        unsigned long long total = 0ULL;
        for (int f = 0; f < BINS; ++f) {
            total += fine[n][f];
        }
        assert(total == powers[n]);
        assert(powers[n] > ZERO_POWER);
    }

    // (b) additivity over streams: each stream alone, then summed.
    unsigned long long fine_sum[TERMS][BINS] = {};
    for (int s = 0; s < TWO_STREAMS; ++s) {
        std::vector<int> one(static_cast<std::size_t>(TERMS) * WINDOWS * 2, 0);
        for (int n = 0; n < TERMS; ++n) {
            for (int m = 0; m < WINDOWS; ++m) {
                const std::size_t src =
                    (static_cast<std::size_t>(n) * rows + s * WINDOWS + m) * 2;
                const std::size_t dst =
                    (static_cast<std::size_t>(n) * WINDOWS + m) * 2;
                one[dst + 0] = row_sums[src + 0];
                one[dst + 1] = row_sums[src + 1];
            }
        }
        unsigned long long fine_one[TERMS][BINS];
        pilotproxy_ref_fine_powers_u64(one.data(), SINGLE_STREAM, fine_one);
        for (int n = 0; n < TERMS; ++n) {
            for (int f = 0; f < BINS; ++f) {
                fine_sum[n][f] += fine_one[n][f];
            }
        }
    }
    for (int n = 0; n < TERMS; ++n) {
        for (int f = 0; f < BINS; ++f) {
            assert(fine_sum[n][f] == fine[n][f]);
        }
    }

    // (c) Parseval to rounding: |P - L_F sum|z|^2| / (L_F sum|z|^2) small.
    for (int n = 0; n < TERMS; ++n) {
        double energy = 0.0;
        for (int m = 0; m < rows; ++m) {
            const std::size_t idx = (static_cast<std::size_t>(n) * rows + m) * 2;
            const double zr = row_sums[idx + 0];
            const double zi = row_sums[idx + 1];
            energy += zr * zr + zi * zi;
        }
        const double parseval = static_cast<double>(BINS) * energy;
        const double drift =
            std::fabs(static_cast<double>(powers[n]) - parseval) / parseval;
        assert(drift <= PARSEVAL_RELATIVE_TOLERANCE);
    }
}

int main()
{
    run_row_sums_reference_checks();
    run_fine_reference_checks();
    assert(pilotproxy_ref_sign_extend_i4(INT4_ZERO_CODE) == INT4_ZERO_VALUE);
    assert(
        pilotproxy_ref_sign_extend_i4(INT4_MAX_POSITIVE_CODE)
        == INT4_MAX_POSITIVE_VALUE);
    assert(
        pilotproxy_ref_sign_extend_i4(INT4_MIN_NEGATIVE_CODE)
        == INT4_MIN_NEGATIVE_VALUE);
    assert(
        pilotproxy_ref_sign_extend_i4(INT4_NEGATIVE_ONE_CODE)
        == INT4_NEGATIVE_ONE_VALUE);

    assert_unpack(pack_i4(-8, 7), -8, 7);
    assert_unpack(pack_i4(1, -1), 1, -1);

    assert_mul(1, 1, 1, 1, 2, 0);
    assert_mul(1, 0, 0, 1, 0, -1);
    assert_mul(0, 1, 1, 0, 0, 1);
    assert_mul(1, -1, 1, 1, 0, -2);

    assert_impulse_fine_powers(0);  // real unit impulse
    assert_impulse_fine_powers(1);  // imaginary unit impulse
    assert_null_arguments_are_harmless();
    assert_frame_power_is_the_all_bin_sum();

    return 0;
}
