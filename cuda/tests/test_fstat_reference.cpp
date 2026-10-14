#include "f_statistic.h"
#include "f_statistic_reference.h"

#include <cassert>
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
constexpr int SINGLE_DETECTOR_ROW = 1;

constexpr unsigned long long ZERO_POWER = 0ULL;
constexpr unsigned long long UNIT_COMPLEX_POWER = 1ULL;
constexpr unsigned long long TWO_COMPONENT_COMPLEX_POWER = 4ULL;
constexpr unsigned long long CHECKED_RATIONAL_SENTINEL_NUM = 123ULL;
constexpr unsigned long long CHECKED_RATIONAL_SENTINEL_DEN = 456ULL;
constexpr unsigned long long UNIT_THRESHOLD_NUM = 1ULL;
constexpr unsigned long long UNIT_THRESHOLD_DEN = 1ULL;
constexpr unsigned long long FULL_THRESHOLD_EXAMPLE_NUM = 1027ULL;
constexpr unsigned long long FULL_THRESHOLD_EXAMPLE_DEN = 1024ULL;
constexpr unsigned long long HALF_THRESHOLD_EXAMPLE_DEN = 2048ULL;
constexpr unsigned long long INVALID_THRESHOLD_DEN = 0ULL;
constexpr unsigned long long OVERFLOW_THRESHOLD_DEN = ~0ULL;
constexpr unsigned long long RAW_F_EXAMPLE_NUM = 3ULL;
constexpr unsigned long long RAW_F_EXAMPLE_DEN = 2ULL;
constexpr double RAW_F_EXAMPLE_VALUE = 3.0;
constexpr double PILOT_EXCESS_EXAMPLE_VALUE = 2.0;
constexpr double PILOT_EXCESS_SENTINEL = 123.0;

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
    fstat_ref_unpack_complex_i4(packed, &real, &imag);
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
    fstat_ref_complex_mul_conj(xr, xi, wr, wi, &real, &imag);
    assert(real == expected_real);
    assert(imag == expected_imag);
}

void assert_single_tap_power(
    InputType x0,
    InputType w0,
    unsigned long long expected_power)
{
    InputType x[FSTAT_DETECTOR_WINDOW_SAMPLES] = {};
    InputType w[FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES] = {};
    x[0] = x0;
    w[0] = w0;

    unsigned long long powers[FSTAT_REFERENCE_POWER_TERMS] = {};
    fstat_ref_powers_u64(x, w, SINGLE_DETECTOR_ROW, powers);
    assert(powers[FSTAT_TARGET_WEIGHT_INDEX] == expected_power);
    assert(powers[FSTAT_LOWER_REFERENCE_WEIGHT_INDEX] == ZERO_POWER);
    assert(powers[FSTAT_UPPER_REFERENCE_WEIGHT_INDEX] == ZERO_POWER);
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
        static_cast<std::size_t>(rows) * FSTAT_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        static_cast<std::size_t>(FSTAT_NUM_WEIGHT_TERMS)
        * FSTAT_DETECTOR_WINDOW_SAMPLES);
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
        static_cast<std::size_t>(FSTAT_NUM_WEIGHT_TERMS) * rows * 2, 0);
    fstat_ref_row_sums_i32(x.data(), w.data(), rows, row_sums.data());

    // (a) exact equality against an independently ordered brute force
    // (term-outer, tap-outer accumulation): integer addition is associative,
    // so any ordering must agree bit-for-bit.
    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        for (int m = 0; m < rows; ++m) {
            long long br = 0;
            long long bi = 0;
            for (int k = 0; k < FSTAT_DETECTOR_WINDOW_SAMPLES; ++k) {
                int xr;
                int xi;
                int wr;
                int wi;
                int yr;
                int yi;
                fstat_ref_unpack_complex_i4(
                    x[static_cast<std::size_t>(m)
                          * FSTAT_DETECTOR_WINDOW_SAMPLES
                      + k],
                    &xr,
                    &xi);
                fstat_ref_unpack_complex_i4(
                    w[static_cast<std::size_t>(n)
                          * FSTAT_DETECTOR_WINDOW_SAMPLES
                      + k],
                    &wr,
                    &wi);
                fstat_ref_complex_mul_conj(xr, xi, wr, wi, &yr, &yi);
                br += yr;
                bi += yi;
            }
            const std::size_t idx =
                (static_cast<std::size_t>(n) * rows + m) * 2;
            assert(static_cast<long long>(row_sums[idx + 0]) == br);
            assert(static_cast<long long>(row_sums[idx + 1]) == bi);
        }
    }

    // (b) exact v1 marginal identity: sum over rows of |z|^2 in int64
    // reproduces fstat_ref_powers_u64 bit-for-bit.
    unsigned long long powers[FSTAT_REFERENCE_POWER_TERMS] = {};
    fstat_ref_powers_u64(x.data(), w.data(), rows, powers);
    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        unsigned long long marginal = 0ULL;
        for (int m = 0; m < rows; ++m) {
            const std::size_t idx =
                (static_cast<std::size_t>(n) * rows + m) * 2;
            const long long zr = row_sums[idx + 0];
            const long long zi = row_sums[idx + 1];
            marginal += static_cast<unsigned long long>(zr * zr + zi * zi);
        }
        assert(marginal == powers[n]);
    }
}

int main()
{
    run_row_sums_reference_checks();
    assert(fstat_ref_sign_extend_i4(INT4_ZERO_CODE) == INT4_ZERO_VALUE);
    assert(
        fstat_ref_sign_extend_i4(INT4_MAX_POSITIVE_CODE)
        == INT4_MAX_POSITIVE_VALUE);
    assert(
        fstat_ref_sign_extend_i4(INT4_MIN_NEGATIVE_CODE)
        == INT4_MIN_NEGATIVE_VALUE);
    assert(
        fstat_ref_sign_extend_i4(INT4_NEGATIVE_ONE_CODE)
        == INT4_NEGATIVE_ONE_VALUE);

    assert_unpack(pack_i4(-8, 7), -8, 7);
    assert_unpack(pack_i4(1, -1), 1, -1);

    assert_mul(1, 1, 1, 1, 2, 0);
    assert_mul(1, 0, 0, 1, 0, -1);
    assert_mul(0, 1, 1, 0, 0, 1);
    assert_mul(1, -1, 1, 1, 0, -2);

    assert_single_tap_power(
        pack_i4(1, 1), pack_i4(1, 1), TWO_COMPONENT_COMPLEX_POWER);
    assert_single_tap_power(
        pack_i4(1, 0), pack_i4(0, 1), UNIT_COMPLEX_POWER);
    assert_single_tap_power(
        pack_i4(0, 1), pack_i4(1, 0), UNIT_COMPLEX_POWER);
    assert_single_tap_power(
        pack_i4(1, -1), pack_i4(1, 1), TWO_COMPONENT_COMPLEX_POWER);

    FStatRational checked = {
        CHECKED_RATIONAL_SENTINEL_NUM,
        CHECKED_RATIONAL_SENTINEL_DEN};
    assert(FStat_MakeHalfThresholdFromFullChecked(
        FULL_THRESHOLD_EXAMPLE_NUM,
        FULL_THRESHOLD_EXAMPLE_DEN,
        &checked) == 1);
    assert(checked.num == FULL_THRESHOLD_EXAMPLE_NUM);
    assert(checked.den == HALF_THRESHOLD_EXAMPLE_DEN);

    assert(FStat_MakeHalfThresholdFromFullChecked(
        UNIT_THRESHOLD_NUM,
        INVALID_THRESHOLD_DEN,
        &checked) == 0);
    assert(checked.num == ZERO_POWER);
    assert(checked.den == ZERO_POWER);

    assert(FStat_MakeHalfThresholdFromFullChecked(
        UNIT_THRESHOLD_NUM,
        OVERFLOW_THRESHOLD_DEN,
        &checked) == 0);
    assert(checked.num == ZERO_POWER);
    assert(checked.den == ZERO_POWER);

    assert(FStat_MakeHalfThresholdFromFullChecked(
        UNIT_THRESHOLD_NUM,
        UNIT_THRESHOLD_DEN,
        0) == 0);
    assert(FStat_NumDenToRawF(RAW_F_EXAMPLE_NUM, RAW_F_EXAMPLE_DEN)
        == RAW_F_EXAMPLE_VALUE);
    assert(FStat_NumDenToRawF(RAW_F_EXAMPLE_NUM, INVALID_THRESHOLD_DEN) == 0.0);
    assert(FStat_NumDenToPilotExcess(RAW_F_EXAMPLE_NUM, RAW_F_EXAMPLE_DEN)
        == PILOT_EXCESS_EXAMPLE_VALUE);
    assert(FStat_NumDenToPilotExcess(RAW_F_EXAMPLE_NUM, INVALID_THRESHOLD_DEN)
        == 0.0);

    double rho = PILOT_EXCESS_SENTINEL;
    assert(FStat_NumDenToPilotExcessChecked(
        RAW_F_EXAMPLE_NUM,
        RAW_F_EXAMPLE_DEN,
        &rho) == 1);
    assert(rho == PILOT_EXCESS_EXAMPLE_VALUE);
    assert(FStat_NumDenToPilotExcessChecked(
        RAW_F_EXAMPLE_NUM,
        INVALID_THRESHOLD_DEN,
        &rho) == 0);
    assert(rho == 0.0);
    assert(FStat_NumDenToPilotExcessChecked(
        RAW_F_EXAMPLE_NUM,
        RAW_F_EXAMPLE_DEN,
        0) == 0);

    return 0;
}
