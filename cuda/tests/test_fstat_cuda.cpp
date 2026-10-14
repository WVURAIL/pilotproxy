#include "f_statistic.h"
#include "f_statistic_reference.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>

namespace {

constexpr int INT4_BITS = 4;
constexpr int INT4_VALUE_COUNT = 16;
constexpr int INT4_SIGNED_MIN = -8;
constexpr int INT4_MASK = 0xF;
constexpr int PACKED_REAL_SHIFT_BITS = 4;

// Numerical Recipes LCG constants; deterministic test data, not cryptography.
constexpr unsigned int LCG_MULTIPLIER = 1664525u;
constexpr unsigned int LCG_INCREMENT = 1013904223u;

constexpr unsigned int ROW_INPUT_SEED_BASE = 1000u;
constexpr unsigned int ROW_WEIGHT_SEED_BASE = 2000u;
constexpr unsigned int CACHE_INPUT_SEED = 3001u;
constexpr unsigned int CACHE_WEIGHT1_SEED = 3002u;
constexpr unsigned int CACHE_WEIGHT2_SEED = 3003u;
constexpr unsigned int BATCH_INPUT_SEED = 4001u;
constexpr unsigned int BATCH_WEIGHT_SEED = 4002u;
constexpr unsigned int STREAM_INPUT_SEED = 5001u;
constexpr unsigned int STREAM_WEIGHT_SEED = 5002u;

constexpr int ROW_COUNT_BELOW_BLOCK_THREADS = FSTAT_BLOCK_THREADS - 1;
constexpr int ROW_COUNT_EQUAL_BLOCK_THREADS = FSTAT_BLOCK_THREADS;
constexpr int ROW_COUNT_ABOVE_BLOCK_THREADS = FSTAT_BLOCK_THREADS + 1;
constexpr int ROW_COUNT_GRID_STRIDE_STRESS = 4101;
constexpr int ROW_COUNT_SMALL_GRID_STRESS = 257;
constexpr int CACHE_REUSE_ROWS = 65;
constexpr int BATCH_EQUIVALENCE_ROWS = 65;
constexpr int BATCH_EQUIVALENCE_COUNT = 3;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(                                                       \
                stderr,                                                         \
                "CUDA error %s:%d: %s\n",                                       \
                __FILE__,                                                       \
                __LINE__,                                                       \
                cudaGetErrorString(err__));                                     \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

InputType pack_i4(int real, int imag)
{
    return static_cast<InputType>(
        ((real & INT4_MASK) << PACKED_REAL_SHIFT_BITS) | (imag & INT4_MASK));
}

unsigned int next_lcg(unsigned int& state)
{
    state = LCG_MULTIPLIER * state + LCG_INCREMENT;
    return state;
}

int random_i4(unsigned int& state)
{
    return static_cast<int>(next_lcg(state) % INT4_VALUE_COUNT) + INT4_SIGNED_MIN;
}

void fill_random_packed(std::vector<InputType>& values, unsigned int seed)
{
    unsigned int state = seed;
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = pack_i4(random_i4(state), random_i4(state));
    }
}

void check_last_error_clear()
{
    const char* err = FStat_LastError();
    assert(err != 0);
    assert(err[0] == '\0');
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : ptr_(0), count_(count)
    {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    ~DeviceBuffer()
    {
        if (ptr_ != 0) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

    void copy_from_host(const std::vector<T>& host)
    {
        assert(host.size() == count_);
        CUDA_CHECK(cudaMemcpy(
            ptr_,
            host.data(),
            count_ * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    void copy_to_host(std::vector<T>& host) const
    {
        host.resize(count_);
        CUDA_CHECK(cudaMemcpy(
            host.data(),
            ptr_,
            count_ * sizeof(T),
            cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_;
    std::size_t count_;
};

std::vector<unsigned long long> cpu_powers(
    const std::vector<InputType>& x,
    const std::vector<InputType>& w,
    int rows)
{
    unsigned long long powers[FSTAT_NUM_WEIGHT_TERMS] = {};
    fstat_ref_powers_u64(x.data(), w.data(), rows, powers);
    return std::vector<unsigned long long>(
        powers,
        powers + FSTAT_NUM_WEIGHT_TERMS);
}

std::vector<unsigned long long> gpu_powers(
    const std::vector<InputType>& x,
    const std::vector<InputType>& w,
    int rows)
{
    DeviceBuffer<InputType> d_x(x.size());
    DeviceBuffer<unsigned long long> d_powers(FSTAT_NUM_WEIGHT_TERMS);
    d_x.copy_from_host(x);

    void* handle = FStat_Create(d_x.get(), 0, rows);
    assert(handle != 0);
    check_last_error_clear();

    FStat_Compute_Powers_U64(handle, w.data(), d_powers.get());
    check_last_error_clear();
    FStat_Destroy(handle);
    check_last_error_clear();

    std::vector<unsigned long long> out;
    d_powers.copy_to_host(out);
    return out;
}

void test_rows_exact()
{
    const int row_counts[] = {
        1,
        ROW_COUNT_BELOW_BLOCK_THREADS,
        ROW_COUNT_EQUAL_BLOCK_THREADS,
        ROW_COUNT_ABOVE_BLOCK_THREADS,
#if FSTAT_GRID_MAX_BLOCKS <= 4
        ROW_COUNT_SMALL_GRID_STRESS,
#endif
        ROW_COUNT_GRID_STRIDE_STRESS};
    for (int rows : row_counts) {
        std::vector<InputType> x(rows * FSTAT_DETECTOR_WINDOW_SAMPLES);
        std::vector<InputType> w(
            FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES);
        fill_random_packed(x, ROW_INPUT_SEED_BASE + static_cast<unsigned int>(rows));
        fill_random_packed(w, ROW_WEIGHT_SEED_BASE + static_cast<unsigned int>(rows));

        assert(gpu_powers(x, w, rows) == cpu_powers(x, w, rows));
    }
}

void test_set_stream_matches_default()
{
    // A handle bound to a non-blocking stream must produce the same powers as the
    // default stream, and switching back must re-upload the weights on the new
    // stream rather than reuse a cache ordered on the old one.
    const int rows = CACHE_REUSE_ROWS;
    std::vector<InputType> x(rows * FSTAT_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES);
    fill_random_packed(x, STREAM_INPUT_SEED);
    fill_random_packed(w, STREAM_WEIGHT_SEED);
    const std::vector<unsigned long long> want = cpu_powers(x, w, rows);

    DeviceBuffer<InputType> d_x(x.size());
    DeviceBuffer<unsigned long long> d_powers(FSTAT_NUM_WEIGHT_TERMS);
    d_x.copy_from_host(x);
    cudaStream_t stream = 0;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    void* handle = FStat_Create(d_x.get(), 0, rows);
    assert(handle != 0);
    FStat_SetStream(handle, stream);
    check_last_error_clear();
    FStat_Compute_Powers_U64(handle, w.data(), d_powers.get());
    check_last_error_clear();
    // A non-blocking stream is not ordered before the synchronous copy below.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned long long> got;
    d_powers.copy_to_host(got);
    assert(got == want);

    FStat_SetStream(handle, 0);
    check_last_error_clear();
    FStat_Compute_Powers_U64(handle, w.data(), d_powers.get());
    check_last_error_clear();
    d_powers.copy_to_host(got);
    assert(got == want);

    FStat_SetStream(0, stream);
    assert(FStat_LastError()[0] != '\0');

    FStat_Destroy(handle);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_weight_cache_reuse_and_change()
{
    const int rows = CACHE_REUSE_ROWS;
    std::vector<InputType> x(rows * FSTAT_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w1(
        FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w2(w1.size());
    fill_random_packed(x, CACHE_INPUT_SEED);
    fill_random_packed(w1, CACHE_WEIGHT1_SEED);
    fill_random_packed(w2, CACHE_WEIGHT2_SEED);

    DeviceBuffer<InputType> d_x(x.size());
    DeviceBuffer<unsigned long long> d_powers(FSTAT_NUM_WEIGHT_TERMS);
    d_x.copy_from_host(x);

    void* handle = FStat_Create(d_x.get(), 0, rows);
    assert(handle != 0);

    std::vector<unsigned long long> got;

    FStat_Compute_Powers_U64(handle, w1.data(), d_powers.get());
    check_last_error_clear();
    d_powers.copy_to_host(got);
    assert(got == cpu_powers(x, w1, rows));

    FStat_Compute_Powers_U64(handle, w1.data(), d_powers.get());
    check_last_error_clear();
    d_powers.copy_to_host(got);
    assert(got == cpu_powers(x, w1, rows));

    FStat_Compute_Powers_U64(handle, w2.data(), d_powers.get());
    check_last_error_clear();
    d_powers.copy_to_host(got);
    assert(got == cpu_powers(x, w2, rows));

    FStat_Destroy(handle);
    check_last_error_clear();
}

void test_batch_equivalence()
{
    const int rows = BATCH_EQUIVALENCE_ROWS;
    const int batch = BATCH_EQUIVALENCE_COUNT;
    std::vector<InputType> x(
        batch * rows * FSTAT_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES);
    fill_random_packed(x, BATCH_INPUT_SEED);
    fill_random_packed(w, BATCH_WEIGHT_SEED);

    DeviceBuffer<InputType> d_x(x.size());
    DeviceBuffer<unsigned long long> d_powers(
        batch * FSTAT_NUM_WEIGHT_TERMS);
    d_x.copy_from_host(x);

    void* handle = FStat_Create_Batch(d_x.get(), 0, rows, batch);
    assert(handle != 0);
    FStat_Compute_Powers_U64(handle, w.data(), d_powers.get());
    check_last_error_clear();
    FStat_Destroy(handle);
    check_last_error_clear();

    std::vector<unsigned long long> got;
    d_powers.copy_to_host(got);

    for (int b = 0; b < batch; ++b) {
        const InputType* begin =
            x.data() + b * rows * FSTAT_DETECTOR_WINDOW_SAMPLES;
        std::vector<InputType> xb(
            begin,
            begin + rows * FSTAT_DETECTOR_WINDOW_SAMPLES);
        std::vector<unsigned long long> expected = cpu_powers(xb, w, rows);
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            assert(got[b * FSTAT_NUM_WEIGHT_TERMS + n] == expected[n]);
        }
    }
}

void test_zero_denominator_and_threshold_equality()
{
    const int rows = 1;
    std::vector<InputType> x(FSTAT_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES);
    x[0] = pack_i4(1, 0);
    w[0] = pack_i4(1, 0);

    DeviceBuffer<InputType> d_x(x.size());
    DeviceBuffer<float> d_f(1);
    DeviceBuffer<unsigned long long> d_num(1);
    DeviceBuffer<unsigned long long> d_den(1);
    DeviceBuffer<unsigned int> d_overflow(1);
    DeviceBuffer<unsigned char> d_mask(1);
    d_x.copy_from_host(x);

    void* handle = FStat_Create(d_x.get(), d_f.get(), rows);
    assert(handle != 0);

    FStat_Compute_DiagnosticFloat(handle, w.data());
    check_last_error_clear();
    std::vector<float> f_out;
    d_f.copy_to_host(f_out);
    assert(f_out[0] == 0.0f);

    FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount(
        handle,
        w.data(),
        1ULL,
        1ULL,
        d_num.get(),
        d_den.get(),
        d_mask.get(),
        d_overflow.get());
    check_last_error_clear();

    std::vector<unsigned long long> num;
    std::vector<unsigned long long> den;
    std::vector<unsigned int> overflow;
    std::vector<unsigned char> mask;
    d_num.copy_to_host(num);
    d_den.copy_to_host(den);
    d_overflow.copy_to_host(overflow);
    d_mask.copy_to_host(mask);
    assert(num[0] == 1ULL);
    assert(den[0] == 0ULL);
    assert(mask[0] == 0u);
    assert(overflow[0] == 0u);

    w[FSTAT_DETECTOR_WINDOW_SAMPLES] = pack_i4(1, 0);
    FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount(
        handle,
        w.data(),
        1ULL,
        1ULL,
        d_num.get(),
        d_den.get(),
        d_mask.get(),
        d_overflow.get());
    check_last_error_clear();

    d_num.copy_to_host(num);
    d_den.copy_to_host(den);
    d_overflow.copy_to_host(overflow);
    d_mask.copy_to_host(mask);
    assert(num[0] == 1ULL);
    assert(den[0] == 1ULL);
    assert(mask[0] == 0u);  /* exact equality is NOT positive excess (strict >) */
    assert(overflow[0] == 0u);

    FStat_Destroy(handle);
    check_last_error_clear();
}

void run_deployed_numden_case(
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned char expected_mask,
    unsigned int expected_overflow)
{
    const int rows = 1;
    std::vector<InputType> x(FSTAT_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES);
    x[0] = pack_i4(1, 0);
    w[0] = pack_i4(1, 0);
    w[FSTAT_DETECTOR_WINDOW_SAMPLES] = pack_i4(1, 0);
    w[2 * FSTAT_DETECTOR_WINDOW_SAMPLES] = pack_i4(1, 0);

    DeviceBuffer<InputType> d_x(x.size());
    DeviceBuffer<unsigned long long> d_num(1);
    DeviceBuffer<unsigned long long> d_den(1);
    DeviceBuffer<unsigned int> d_overflow(1);
    DeviceBuffer<unsigned char> d_mask(1);
    d_x.copy_from_host(x);

    void* handle = FStat_Create(d_x.get(), 0, rows);
    assert(handle != 0);

    FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount(
        handle,
        w.data(),
        threshold_half_num,
        threshold_half_den,
        d_num.get(),
        d_den.get(),
        d_mask.get(),
        d_overflow.get());
    check_last_error_clear();
    FStat_Destroy(handle);
    check_last_error_clear();

    std::vector<unsigned long long> num;
    std::vector<unsigned long long> den;
    std::vector<unsigned int> overflow;
    std::vector<unsigned char> mask;
    d_num.copy_to_host(num);
    d_den.copy_to_host(den);
    d_overflow.copy_to_host(overflow);
    d_mask.copy_to_host(mask);

    assert(num[0] == 1ULL);
    assert(den[0] == 2ULL);
    assert(FStat_NumDenToRawF(num[0], den[0]) == 1.0);
    assert(mask[0] == expected_mask);
    if (expected_overflow == 0u) {
        assert(overflow[0] == 0u);
    } else {
        assert(overflow[0] >= expected_overflow);
    }
}

void test_deployed_numden_null_output_threshold_cases()
{
    run_deployed_numden_case(3ULL, 1ULL, 0u, 0u);
    run_deployed_numden_case(1ULL, 2ULL, 0u, 0u);  /* 1*2 == 1*2: equality masks 0 */
    run_deployed_numden_case(1ULL, 3ULL, 1u, 0u);  /* 1*3 > 1*2: just-above boundary */
    run_deployed_numden_case(1ULL, 4ULL, 1u, 0u);
}

void test_rational_overflow_telemetry()
{
    run_deployed_numden_case(~0ULL, 1ULL, 0u, 1u);
}

}  // namespace

static std::vector<int> cpu_row_sums(
    const std::vector<InputType>& x,
    const std::vector<InputType>& w,
    int rows)
{
    std::vector<int> out(
        static_cast<std::size_t>(FSTAT_NUM_WEIGHT_TERMS) * rows * 2, 0);
    fstat_ref_row_sums_i32(x.data(), w.data(), rows, out.data());
    return out;
}

// v2 front end: GPU row sums must match the CPU reference EXACTLY (integer
// outputs, no tolerance), and the int64 marginal over rows must reproduce
// the v1 uint64 powers bit-for-bit.
static void test_row_sums_exact()
{
    const int row_counts[] = {1, 3, 64, 257};
    for (int rows : row_counts) {
        std::vector<InputType> x(
            static_cast<std::size_t>(rows) * FSTAT_DETECTOR_WINDOW_SAMPLES);
        std::vector<InputType> w(
            static_cast<std::size_t>(FSTAT_NUM_WEIGHT_TERMS)
            * FSTAT_DETECTOR_WINDOW_SAMPLES);
        fill_random_packed(x, ROW_INPUT_SEED_BASE + 7000u + static_cast<unsigned int>(rows));
        fill_random_packed(w, ROW_WEIGHT_SEED_BASE + 7000u + static_cast<unsigned int>(rows));

        DeviceBuffer<InputType> d_x(x.size());
        d_x.copy_from_host(x);
        DeviceBuffer<int> d_row_sums(
            static_cast<std::size_t>(FSTAT_NUM_WEIGHT_TERMS) * rows * 2);

        void* handle = FStat_Create(d_x.get(), 0, rows);
        assert(handle != 0);
        assert(FStat_Supports_RowSums() == 1);
        FStat_Compute_RowSums_I32(handle, w.data(), d_row_sums.get());
        check_last_error_clear();

        std::vector<int> got;
        d_row_sums.copy_to_host(got);
        const std::vector<int> expected = cpu_row_sums(x, w, rows);
        assert(got == expected);

        // exact v1 marginal identity against the deployed powers path
        DeviceBuffer<unsigned long long> d_powers(FSTAT_NUM_WEIGHT_TERMS);
        FStat_Compute_Powers_U64(handle, w.data(), d_powers.get());
        check_last_error_clear();
        std::vector<unsigned long long> powers;
        d_powers.copy_to_host(powers);
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            unsigned long long marginal = 0ULL;
            for (int m = 0; m < rows; ++m) {
                const std::size_t idx =
                    (static_cast<std::size_t>(n) * rows + m) * 2;
                const long long zr = got[idx + 0];
                const long long zi = got[idx + 1];
                marginal += static_cast<unsigned long long>(zr * zr + zi * zi);
            }
            assert(marginal == powers[n]);
        }

        FStat_Destroy(handle);
    }
}

// Batched v2 front end: each batch entry must equal the single-block
// reference on its own input slice, exactly.
static void test_row_sums_batch_exact()
{
    const int rows = 96;
    const int batch = 5;
    std::vector<InputType> x(
        static_cast<std::size_t>(batch) * rows
        * FSTAT_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        static_cast<std::size_t>(FSTAT_NUM_WEIGHT_TERMS)
        * FSTAT_DETECTOR_WINDOW_SAMPLES);
    fill_random_packed(x, ROW_INPUT_SEED_BASE + 9100u);
    fill_random_packed(w, ROW_WEIGHT_SEED_BASE + 9100u);

    DeviceBuffer<InputType> d_x(x.size());
    d_x.copy_from_host(x);
    DeviceBuffer<int> d_row_sums(
        static_cast<std::size_t>(batch) * FSTAT_NUM_WEIGHT_TERMS * rows * 2);

    void* handle = FStat_Create_Batch(d_x.get(), 0, rows, batch);
    assert(handle != 0);
    FStat_Compute_RowSums_I32(handle, w.data(), d_row_sums.get());
    check_last_error_clear();

    std::vector<int> got;
    d_row_sums.copy_to_host(got);

    const std::size_t block_in =
        static_cast<std::size_t>(rows) * FSTAT_DETECTOR_WINDOW_SAMPLES;
    const std::size_t block_out =
        static_cast<std::size_t>(FSTAT_NUM_WEIGHT_TERMS) * rows * 2;
    for (int b = 0; b < batch; ++b) {
        std::vector<InputType> xb(
            x.begin() + b * block_in,
            x.begin() + (b + 1) * block_in);
        const std::vector<int> expected = cpu_row_sums(xb, w, rows);
        for (std::size_t i = 0; i < block_out; ++i) {
            assert(got[b * block_out + i] == expected[i]);
        }
    }

    FStat_Destroy(handle);
}

int main()
{
    int major = 0;
    int minor = 0;
    int patch = 0;
    FStat_GetVersion(&major, &minor, &patch);
    // Guard library <-> source consistency, not a literal: a stale
    // libfstatistic.so must fail here before any semantic test runs.
    assert(major == FSTAT_CORE_VERSION_MAJOR);
    assert(minor == FSTAT_CORE_VERSION_MINOR);
    assert(patch == FSTAT_CORE_VERSION_PATCH);

    test_rows_exact();
    test_row_sums_exact();
    test_row_sums_batch_exact();
    test_weight_cache_reuse_and_change();
    test_set_stream_matches_default();
    test_batch_equivalence();
    test_zero_denominator_and_threshold_equality();
    test_deployed_numden_null_output_threshold_cases();
    test_rational_overflow_telemetry();

    return 0;
}
