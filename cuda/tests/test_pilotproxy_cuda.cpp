#include "pilotproxy_detector.h"
#include "pilotproxy_reference.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>

// CUDA-vs-CPU regression for kernel core 3.0.0. Every power the library
// emits is exact, so every comparison here is bit equality:
//   * row sums (PilotProxy_Compute_RowSums_I32) == pilotproxy_ref_row_sums_i32;
//   * fused fine powers (PilotProxy_Compute_FusedFine_U64) == the frozen
//     fxfft256 v1 reference over the CPU row sums;
//   * frame powers d_power_out == the host sum over bins of
//     d_fine_power_out == pilotproxy_ref_powers_from_fine_u64 (the kernel core
//     3.0.0 definition), on every fused entry that can bind the tap.

namespace {

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
constexpr unsigned int MASK_INPUT_SEED = 6001u;
constexpr unsigned int MASK_WEIGHT_SEED = 6002u;

constexpr int TERMS = PILOTPROXY_NUM_WEIGHT_TERMS;
constexpr int WINDOWS = PILOTPROXY_FINE_WINDOWS_PER_STREAM;
constexpr int BINS = PILOTPROXY_FINE_NUM_BINS;

constexpr int ROW_COUNT_BELOW_BLOCK_THREADS = PILOTPROXY_BLOCK_THREADS - 1;
constexpr int ROW_COUNT_EQUAL_BLOCK_THREADS = PILOTPROXY_BLOCK_THREADS;
constexpr int ROW_COUNT_ABOVE_BLOCK_THREADS = PILOTPROXY_BLOCK_THREADS + 1;
constexpr int ROW_COUNT_GRID_STRIDE_STRESS = 4101;
constexpr int ROW_COUNT_SMALL_GRID_STRESS = 257;
constexpr int CACHE_REUSE_ROWS = WINDOWS;
constexpr int BATCH_EQUIVALENCE_ROWS = 2 * WINDOWS;
constexpr int BATCH_EQUIVALENCE_COUNT = 3;
constexpr int MASK_ROWS = 2 * WINDOWS;
constexpr int MASK_BATCH = 2;
constexpr unsigned long long UNIT_MULTIPLIER_Q16 = 1ULL << 16;
constexpr unsigned long long SENTINEL_U64 = 0xABABABABABABABABULL;

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
    const char* err = PilotProxy_LastError();
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

    void fill_bytes(int value)
    {
        CUDA_CHECK(cudaMemset(ptr_, value, count_ * sizeof(T)));
    }

private:
    T* ptr_;
    std::size_t count_;
};

std::vector<int> cpu_row_sums(
    const std::vector<InputType>& x,
    const std::vector<InputType>& w,
    int rows)
{
    std::vector<int> out(static_cast<std::size_t>(TERMS) * rows * 2, 0);
    pilotproxy_ref_row_sums_i32(x.data(), w.data(), rows, out.data());
    return out;
}

struct FineAndPowers {
    std::vector<unsigned long long> fine;    // [batch][TERMS][BINS]
    std::vector<unsigned long long> powers;  // [batch][TERMS]
};

// The kernel core 3.0.0 definition of the frame powers, applied on the host.
std::vector<unsigned long long> host_sum_over_bins(
    const std::vector<unsigned long long>& fine)
{
    assert(fine.size() % BINS == 0);
    std::vector<unsigned long long> out(fine.size() / BINS, 0ULL);
    for (std::size_t idx = 0; idx < out.size(); ++idx) {
        for (int f = 0; f < BINS; ++f) {
            out[idx] += fine[idx * BINS + f];
        }
    }
    return out;
}

// CPU reference chain for one block: row sums -> fxfft256 v1 fine powers ->
// frame powers.
FineAndPowers cpu_fine_and_powers(
    const std::vector<InputType>& x,
    const std::vector<InputType>& w,
    int rows)
{
    assert(rows % WINDOWS == 0);
    const std::vector<int> row_sums = cpu_row_sums(x, w, rows);
    unsigned long long fine[TERMS][BINS];
    unsigned long long powers[TERMS];
    pilotproxy_ref_fine_powers_u64(row_sums.data(), rows / WINDOWS, fine);
    pilotproxy_ref_powers_from_fine_u64(fine, powers);
    FineAndPowers out;
    out.fine.assign(&fine[0][0], &fine[0][0] + TERMS * BINS);
    out.powers.assign(powers, powers + TERMS);
    return out;
}

FineAndPowers gpu_fused(
    void* handle,
    const std::vector<InputType>& w,
    int rows,
    int batch,
    std::vector<int>* tap_out)
{
    DeviceBuffer<unsigned long long> d_fine(
        static_cast<std::size_t>(batch) * TERMS * BINS);
    DeviceBuffer<unsigned long long> d_powers(
        static_cast<std::size_t>(batch) * TERMS);
    DeviceBuffer<int> d_tap(
        static_cast<std::size_t>(batch) * TERMS * rows * 2);
    d_powers.fill_bytes(0xAB);
    PilotProxy_Compute_FusedFine_U64(
        handle, w.data(), d_fine.get(), d_powers.get(),
        tap_out != 0 ? d_tap.get() : 0);
    check_last_error_clear();
    FineAndPowers out;
    d_fine.copy_to_host(out.fine);
    d_powers.copy_to_host(out.powers);
    if (tap_out != 0) {
        d_tap.copy_to_host(*tap_out);
    }
    return out;
}

void assert_frame_power_identity(const FineAndPowers& gpu)
{
    assert(gpu.powers == host_sum_over_bins(gpu.fine));
}

void assert_fused_matches_cpu(
    const FineAndPowers& gpu,
    const FineAndPowers& cpu,
    int batch_index)
{
    for (int n = 0; n < TERMS; ++n) {
        assert(gpu.powers[static_cast<std::size_t>(batch_index) * TERMS + n]
               == cpu.powers[n]);
        for (int f = 0; f < BINS; ++f) {
            assert(gpu.fine[(static_cast<std::size_t>(batch_index) * TERMS + n) * BINS + f]
                   == cpu.fine[static_cast<std::size_t>(n) * BINS + f]);
        }
    }
}

// Fused fine powers and frame powers against the CPU reference chain for a
// range of stream counts, with the row-sum tap bound.
void test_fused_fine_exact()
{
    const int row_counts[] = {WINDOWS, 2 * WINDOWS, 8 * WINDOWS, 33 * WINDOWS};
    for (int rows : row_counts) {
        std::vector<InputType> x(
            static_cast<std::size_t>(rows) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
        std::vector<InputType> w(
            static_cast<std::size_t>(TERMS) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
        fill_random_packed(x, ROW_INPUT_SEED_BASE + static_cast<unsigned int>(rows));
        fill_random_packed(w, ROW_WEIGHT_SEED_BASE + static_cast<unsigned int>(rows));

        DeviceBuffer<InputType> d_x(x.size());
        d_x.copy_from_host(x);
        void* handle = PilotProxy_Create(d_x.get(), rows);
        assert(handle != 0);
        check_last_error_clear();
        assert(PilotProxy_Supports_FusedFine() == 1);

        std::vector<int> tap;
        const FineAndPowers gpu = gpu_fused(handle, w, rows, 1, &tap);
        PilotProxy_Destroy(handle);
        check_last_error_clear();

        const FineAndPowers cpu = cpu_fine_and_powers(x, w, rows);
        assert(tap == cpu_row_sums(x, w, rows));
        assert_fused_matches_cpu(gpu, cpu, 0);
        assert_frame_power_identity(gpu);
        for (int n = 0; n < TERMS; ++n) {
            assert(gpu.powers[n] > 0ULL);
        }
    }
}

// A unit impulse in window 0 against a unit target weight gives z = (1, 0)
// in one row and zero elsewhere; the frozen transform of that impulse is
// exactly one in every bin, so the fused kernel must emit S_target[f] = 1,
// P_target = 256 and zero for both reference terms --- exactness of the
// device transform and of the frame-power reduction without any reference.
void test_fused_impulse_and_zero_reference_terms()
{
    const int rows = WINDOWS;
    std::vector<InputType> x(
        static_cast<std::size_t>(rows) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES, 0);
    std::vector<InputType> w(
        static_cast<std::size_t>(TERMS) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES, 0);
    x[0] = pack_i4(1, 0);
    w[0] = pack_i4(1, 0);

    DeviceBuffer<InputType> d_x(x.size());
    d_x.copy_from_host(x);
    void* handle = PilotProxy_Create(d_x.get(), rows);
    assert(handle != 0);
    const FineAndPowers gpu = gpu_fused(handle, w, rows, 1, 0);
    PilotProxy_Destroy(handle);
    check_last_error_clear();

    for (int f = 0; f < BINS; ++f) {
        assert(gpu.fine[PILOTPROXY_TARGET_WEIGHT_INDEX * BINS + f] == 1ULL);
        assert(gpu.fine[PILOTPROXY_LOWER_REFERENCE_WEIGHT_INDEX * BINS + f] == 0ULL);
        assert(gpu.fine[PILOTPROXY_UPPER_REFERENCE_WEIGHT_INDEX * BINS + f] == 0ULL);
    }
    assert(gpu.powers[PILOTPROXY_TARGET_WEIGHT_INDEX] == static_cast<unsigned long long>(BINS));
    assert(gpu.powers[PILOTPROXY_LOWER_REFERENCE_WEIGHT_INDEX] == 0ULL);
    assert(gpu.powers[PILOTPROXY_UPPER_REFERENCE_WEIGHT_INDEX] == 0ULL);
    assert_frame_power_identity(gpu);
}

// The fused entries require whole 128-window streams; anything else is an
// API error recorded in PilotProxy_LastError, not a launch.
void test_fused_rejects_partial_streams()
{
    const int rows = WINDOWS + 1;
    std::vector<InputType> x(
        static_cast<std::size_t>(rows) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES, 0);
    std::vector<InputType> w(
        static_cast<std::size_t>(TERMS) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES, 0);
    DeviceBuffer<InputType> d_x(x.size());
    d_x.copy_from_host(x);
    DeviceBuffer<unsigned long long> d_fine(static_cast<std::size_t>(TERMS) * BINS);
    DeviceBuffer<unsigned long long> d_powers(TERMS);
    void* handle = PilotProxy_Create(d_x.get(), rows);
    assert(handle != 0);
    PilotProxy_Compute_FusedFine_U64(handle, w.data(), d_fine.get(), d_powers.get(), 0);
    assert(PilotProxy_LastError()[0] != '\0');
    PilotProxy_Compute_FusedFine_U64(handle, w.data(), d_fine.get(), 0, 0);
    assert(PilotProxy_LastError()[0] != '\0');
    PilotProxy_Destroy(handle);
    check_last_error_clear();
}

void test_set_stream_matches_default()
{
    // A handle bound to a non-blocking stream must produce the same fine and
    // frame powers as the default stream, and switching back must re-upload
    // the weights on the new stream rather than reuse a cache ordered on the
    // old one.
    const int rows = CACHE_REUSE_ROWS;
    std::vector<InputType> x(rows * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(TERMS * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    fill_random_packed(x, STREAM_INPUT_SEED);
    fill_random_packed(w, STREAM_WEIGHT_SEED);
    const FineAndPowers want = cpu_fine_and_powers(x, w, rows);

    DeviceBuffer<InputType> d_x(x.size());
    DeviceBuffer<unsigned long long> d_fine(static_cast<std::size_t>(TERMS) * BINS);
    DeviceBuffer<unsigned long long> d_powers(TERMS);
    d_x.copy_from_host(x);
    cudaStream_t stream = 0;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    void* handle = PilotProxy_Create(d_x.get(), rows);
    assert(handle != 0);
    PilotProxy_SetStream(handle, stream);
    check_last_error_clear();
    PilotProxy_Compute_FusedFine_U64(handle, w.data(), d_fine.get(), d_powers.get(), 0);
    check_last_error_clear();
    // A non-blocking stream is not ordered before the synchronous copy below.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    FineAndPowers got;
    d_fine.copy_to_host(got.fine);
    d_powers.copy_to_host(got.powers);
    assert(got.fine == want.fine);
    assert(got.powers == want.powers);

    PilotProxy_SetStream(handle, 0);
    check_last_error_clear();
    d_powers.fill_bytes(0xAB);
    PilotProxy_Compute_FusedFine_U64(handle, w.data(), d_fine.get(), d_powers.get(), 0);
    check_last_error_clear();
    d_fine.copy_to_host(got.fine);
    d_powers.copy_to_host(got.powers);
    assert(got.fine == want.fine);
    assert(got.powers == want.powers);

    PilotProxy_SetStream(0, stream);
    assert(PilotProxy_LastError()[0] != '\0');

    PilotProxy_Destroy(handle);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_weight_cache_reuse_and_change()
{
    const int rows = CACHE_REUSE_ROWS;
    std::vector<InputType> x(rows * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w1(TERMS * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w2(w1.size());
    fill_random_packed(x, CACHE_INPUT_SEED);
    fill_random_packed(w1, CACHE_WEIGHT1_SEED);
    fill_random_packed(w2, CACHE_WEIGHT2_SEED);

    DeviceBuffer<InputType> d_x(x.size());
    d_x.copy_from_host(x);

    void* handle = PilotProxy_Create(d_x.get(), rows);
    assert(handle != 0);

    const FineAndPowers want1 = cpu_fine_and_powers(x, w1, rows);
    const FineAndPowers want2 = cpu_fine_and_powers(x, w2, rows);
    assert(want1.powers != want2.powers);

    FineAndPowers got = gpu_fused(handle, w1, rows, 1, 0);
    assert(got.fine == want1.fine && got.powers == want1.powers);
    got = gpu_fused(handle, w1, rows, 1, 0);  // cache hit
    assert(got.fine == want1.fine && got.powers == want1.powers);
    got = gpu_fused(handle, w2, rows, 1, 0);  // cache invalidated
    assert(got.fine == want2.fine && got.powers == want2.powers);
    // The host buffer can change without its address changing.
    w1 = w2;
    got = gpu_fused(handle, w1, rows, 1, 0);
    assert(got.fine == want2.fine && got.powers == want2.powers);

    PilotProxy_Destroy(handle);
    check_last_error_clear();
}

void test_batch_equivalence()
{
    const int rows = BATCH_EQUIVALENCE_ROWS;
    const int batch = BATCH_EQUIVALENCE_COUNT;
    std::vector<InputType> x(
        static_cast<std::size_t>(batch) * rows * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(TERMS * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    fill_random_packed(x, BATCH_INPUT_SEED);
    fill_random_packed(w, BATCH_WEIGHT_SEED);

    DeviceBuffer<InputType> d_x(x.size());
    d_x.copy_from_host(x);

    void* handle = PilotProxy_Create_Batch(d_x.get(), rows, batch);
    assert(handle != 0);
    std::vector<int> tap;
    const FineAndPowers got = gpu_fused(handle, w, rows, batch, &tap);
    PilotProxy_Destroy(handle);
    check_last_error_clear();

    assert_frame_power_identity(got);
    const std::size_t block_in =
        static_cast<std::size_t>(rows) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES;
    const std::size_t block_tap = static_cast<std::size_t>(TERMS) * rows * 2;
    for (int b = 0; b < batch; ++b) {
        std::vector<InputType> xb(
            x.begin() + b * block_in, x.begin() + (b + 1) * block_in);
        const FineAndPowers expected = cpu_fine_and_powers(xb, w, rows);
        assert_fused_matches_cpu(got, expected, b);
        const std::vector<int> rows_b = cpu_row_sums(xb, w, rows);
        for (std::size_t i = 0; i < block_tap; ++i) {
            assert(tap[b * block_tap + i] == rows_b[i]);
        }
    }
}

// Both mask entries share the frame-power reduction: with the optional tap
// bound they report the same P_r as PilotProxy_Compute_FusedFine_U64 (the host
// sum over bins of their own fine accumulator); with it unbound the buffer
// is not touched at all.
void test_mask_entries_share_the_frame_power_reduction()
{
    const int rows = MASK_ROWS;
    const int batch = MASK_BATCH;
    std::vector<InputType> x(
        static_cast<std::size_t>(batch) * rows * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(TERMS * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    fill_random_packed(x, MASK_INPUT_SEED);
    fill_random_packed(w, MASK_WEIGHT_SEED);

    DeviceBuffer<InputType> d_x(x.size());
    d_x.copy_from_host(x);
    void* handle = PilotProxy_Create_Batch(d_x.get(), rows, batch);
    assert(handle != 0);

    const FineAndPowers fused = gpu_fused(handle, w, rows, batch, 0);
    assert_frame_power_identity(fused);

    const unsigned long long bulk_words[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    DeviceBuffer<unsigned long long> d_fine(
        static_cast<std::size_t>(batch) * TERMS * BINS);
    DeviceBuffer<unsigned long long> d_powers(static_cast<std::size_t>(batch) * TERMS);
    DeviceBuffer<int> d_mask(batch);
    DeviceBuffer<int> d_rank_valid(batch);
    DeviceBuffer<int> d_n_bulk(batch);
    std::vector<unsigned long long> fine;
    std::vector<unsigned long long> powers;
    std::vector<int> mask;

    for (int with_support = 0; with_support < 2; ++with_support) {
        for (int bind_tap = 1; bind_tap >= 0; --bind_tap) {
            d_powers.fill_bytes(0xAB);
            unsigned long long* tap = bind_tap ? d_powers.get() : 0;
            if (with_support) {
                PilotProxy_Compute_FusedFineMaskWithSupport_U64(
                    handle, w.data(), 0, 0, bulk_words, 0, UNIT_MULTIPLIER_Q16,
                    d_fine.get(), d_mask.get(), d_rank_valid.get(), d_n_bulk.get(),
                    tap, 0);
            } else {
                PilotProxy_Compute_FusedFineMask_U64(
                    handle, w.data(), 0, 0, bulk_words, 0, UNIT_MULTIPLIER_Q16,
                    d_fine.get(), d_mask.get(), tap, 0);
            }
            check_last_error_clear();
            d_fine.copy_to_host(fine);
            d_powers.copy_to_host(powers);
            d_mask.copy_to_host(mask);
            assert(fine == fused.fine);
            for (int b = 0; b < batch; ++b) {
                assert(mask[b] == 0 || mask[b] == 1);
            }
            if (bind_tap) {
                assert(powers == fused.powers);
                assert(powers == host_sum_over_bins(fine));
            } else {
                for (std::size_t i = 0; i < powers.size(); ++i) {
                    assert(powers[i] == SENTINEL_U64);
                }
            }
        }
    }

    PilotProxy_Destroy(handle);
    check_last_error_clear();
}

}  // namespace

// v2 front end: GPU row sums must match the CPU reference EXACTLY (integer
// outputs, no tolerance) across the grid-stride row counts.
static void test_row_sums_exact()
{
    const int row_counts[] = {
        1,
        3,
        ROW_COUNT_BELOW_BLOCK_THREADS,
        ROW_COUNT_EQUAL_BLOCK_THREADS,
        ROW_COUNT_ABOVE_BLOCK_THREADS,
#if PILOTPROXY_GRID_MAX_BLOCKS <= 4
        ROW_COUNT_SMALL_GRID_STRESS,
#endif
        ROW_COUNT_GRID_STRIDE_STRESS};
    for (int rows : row_counts) {
        std::vector<InputType> x(
            static_cast<std::size_t>(rows) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
        std::vector<InputType> w(
            static_cast<std::size_t>(TERMS) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
        fill_random_packed(x, ROW_INPUT_SEED_BASE + 7000u + static_cast<unsigned int>(rows));
        fill_random_packed(w, ROW_WEIGHT_SEED_BASE + 7000u + static_cast<unsigned int>(rows));

        DeviceBuffer<InputType> d_x(x.size());
        d_x.copy_from_host(x);
        DeviceBuffer<int> d_row_sums(static_cast<std::size_t>(TERMS) * rows * 2);

        void* handle = PilotProxy_Create(d_x.get(), rows);
        assert(handle != 0);
        assert(PilotProxy_Supports_RowSums() == 1);
        PilotProxy_Compute_RowSums_I32(handle, w.data(), d_row_sums.get());
        check_last_error_clear();

        std::vector<int> got;
        d_row_sums.copy_to_host(got);
        assert(got == cpu_row_sums(x, w, rows));

        PilotProxy_Destroy(handle);
    }
}

// Batched v2 front end: each batch entry must equal the single-block
// reference on its own input slice, exactly.
static void test_row_sums_batch_exact()
{
    const int rows = 96;
    const int batch = 5;
    std::vector<InputType> x(
        static_cast<std::size_t>(batch) * rows * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    std::vector<InputType> w(
        static_cast<std::size_t>(TERMS) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES);
    fill_random_packed(x, ROW_INPUT_SEED_BASE + 9100u);
    fill_random_packed(w, ROW_WEIGHT_SEED_BASE + 9100u);

    DeviceBuffer<InputType> d_x(x.size());
    d_x.copy_from_host(x);
    DeviceBuffer<int> d_row_sums(
        static_cast<std::size_t>(batch) * TERMS * rows * 2);

    void* handle = PilotProxy_Create_Batch(d_x.get(), rows, batch);
    assert(handle != 0);
    PilotProxy_Compute_RowSums_I32(handle, w.data(), d_row_sums.get());
    check_last_error_clear();

    std::vector<int> got;
    d_row_sums.copy_to_host(got);

    const std::size_t block_in =
        static_cast<std::size_t>(rows) * PILOTPROXY_DETECTOR_WINDOW_SAMPLES;
    const std::size_t block_out = static_cast<std::size_t>(TERMS) * rows * 2;
    for (int b = 0; b < batch; ++b) {
        std::vector<InputType> xb(
            x.begin() + b * block_in,
            x.begin() + (b + 1) * block_in);
        const std::vector<int> expected = cpu_row_sums(xb, w, rows);
        for (std::size_t i = 0; i < block_out; ++i) {
            assert(got[b * block_out + i] == expected[i]);
        }
    }

    PilotProxy_Destroy(handle);
}

int main()
{
    int major = 0;
    int minor = 0;
    int patch = 0;
    PilotProxy_GetVersion(&major, &minor, &patch);
    // Guard library <-> source consistency, not a literal: a stale
    // libpilotproxy.so must fail here before any semantic test runs.
    assert(major == PILOTPROXY_CORE_VERSION_MAJOR);
    assert(minor == PILOTPROXY_CORE_VERSION_MINOR);
    assert(patch == PILOTPROXY_CORE_VERSION_PATCH);

    int windows = 0;
    int pad = 0;
    int bins = 0;
    PilotProxy_GetFineSpecs(&windows, &pad, &bins);
    assert(windows == WINDOWS);
    assert(bins == BINS);

    test_row_sums_exact();
    test_row_sums_batch_exact();
    test_fused_fine_exact();
    test_fused_impulse_and_zero_reference_terms();
    test_fused_rejects_partial_streams();
    test_weight_cache_reuse_and_change();
    test_set_stream_matches_default();
    test_batch_equivalence();
    test_mask_entries_share_the_frame_power_reduction();

    std::printf("test_pilotproxy_cuda: kernel core %d.%d.%d, all checks passed\n",
                major, minor, patch);
    return 0;
}
