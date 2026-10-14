#include <lime/LimeSuite.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kChunkSamples = 262'144;
constexpr unsigned kTimeoutMs = 2'000;

struct Options {
    std::string serial;
    double frequency_hz = 0.0;
    double sample_rate_hz = 0.0;
    double bandwidth_hz = 0.0;
    unsigned tx_gain_db = 0;
    unsigned rx_gain_db = 0;
    std::uint32_t fifo_samples = 0;
    std::uint64_t settle_samples = 0;
    std::uint64_t capture_samples = 0;
    std::uint64_t session_samples = 0;
    std::uint64_t start_delay_samples = 0;
    std::filesystem::path tx_path;
    std::filesystem::path session_rx_path;
    std::filesystem::path tx_off_rx_path;
    std::filesystem::path status_path;
};

struct Counts {
    std::uint64_t underrun = 0;
    std::uint64_t overrun = 0;
    std::uint64_t dropped = 0;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check(int result, const std::string& operation) {
    if (result != LMS_SUCCESS) {
        const char* detail = LMS_GetLastErrorMessage();
        fail(operation + " failed: " + (detail == nullptr ? "unknown error" : detail));
    }
}

std::uint64_t parse_u64(const std::string& value, const std::string& name) {
    std::size_t used = 0;
    const auto parsed = std::stoull(value, &used);
    if (used != value.size()) {
        fail("Invalid " + name + ".");
    }
    return parsed;
}

double parse_double(const std::string& value, const std::string& name) {
    std::size_t used = 0;
    const auto parsed = std::stod(value, &used);
    if (used != value.size()) {
        fail("Invalid " + name + ".");
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    if (argc < 2 || argc % 2 == 0) {
        fail("Expected name-value arguments.");
    }
    std::unordered_map<std::string, std::string> values;
    for (int index = 1; index < argc; index += 2) {
        values.emplace(argv[index], argv[index + 1]);
    }
    const auto get = [&values](const std::string& name) -> const std::string& {
        const auto found = values.find(name);
        if (found == values.end() || found->second.empty()) {
            fail("Missing " + name + ".");
        }
        return found->second;
    };

    Options options;
    options.serial = get("--serial");
    options.frequency_hz = parse_double(get("--frequency-hz"), "frequency");
    options.sample_rate_hz = parse_double(get("--sample-rate-hz"), "sample rate");
    options.bandwidth_hz = parse_double(get("--bandwidth-hz"), "bandwidth");
    options.tx_gain_db = static_cast<unsigned>(parse_u64(get("--tx-gain-db"), "TX gain"));
    options.rx_gain_db = static_cast<unsigned>(parse_u64(get("--rx-gain-db"), "RX gain"));
    options.fifo_samples = static_cast<std::uint32_t>(
        parse_u64(get("--fifo-samples"), "FIFO size"));
    options.settle_samples = parse_u64(get("--settle-samples"), "settle samples");
    options.capture_samples = parse_u64(get("--capture-samples"), "capture samples");
    options.session_samples = parse_u64(get("--session-samples"), "session samples");
    options.start_delay_samples = parse_u64(
        get("--start-delay-samples"), "start delay");
    options.tx_path = get("--tx-file");
    options.session_rx_path = get("--session-rx-file");
    options.tx_off_rx_path = get("--tx-off-rx-file");
    options.status_path = get("--status-file");
    return options;
}

class Device {
public:
    explicit Device(const std::string& serial) {
        const int count = LMS_GetDeviceList(nullptr);
        if (count <= 0) {
            fail("No LimeSDR device found.");
        }
        auto devices = std::make_unique<lms_info_str_t[]>(static_cast<std::size_t>(count));
        const int listed = LMS_GetDeviceList(devices.get());
        if (listed != count) {
            fail("LMS_GetDeviceList returned an inconsistent device count.");
        }
        const char* selected = nullptr;
        for (int index = 0; index < count; ++index) {
            const std::string descriptor(devices[index]);
            const std::string key = "serial=";
            const auto start = descriptor.find(key);
            const auto stop = descriptor.find(',', start);
            const auto found = start == std::string::npos
                ? std::string()
                : descriptor.substr(start + key.size(), stop - start - key.size());
            if (found == serial) {
                selected = devices[index];
                break;
            }
        }
        if (selected == nullptr) {
            fail("Requested LimeSDR serial was not found.");
        }
        check(LMS_Open(&value_, selected, nullptr), "LMS_Open");
        check(LMS_Init(value_), "LMS_Init");
    }

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    ~Device() {
        if (value_ != nullptr) {
            LMS_SetAntenna(value_, LMS_CH_TX, 0, LMS_PATH_NONE);
            LMS_SetGaindB(value_, LMS_CH_TX, 0, 0);
            LMS_EnableChannel(value_, LMS_CH_TX, 0, false);
            LMS_EnableChannel(value_, LMS_CH_RX, 0, false);
            LMS_Close(value_);
        }
    }

    lms_device_t* get() const { return value_; }

private:
    lms_device_t* value_ = nullptr;
};

void silence_tx(lms_device_t* device) noexcept {
    LMS_SetAntenna(device, LMS_CH_TX, 0, LMS_PATH_NONE);
    LMS_SetGaindB(device, LMS_CH_TX, 0, 0);
    LMS_EnableChannel(device, LMS_CH_TX, 0, false);
}

class Stream {
public:
    Stream(lms_device_t* device, bool tx, std::uint32_t fifo_samples)
        : device_(device) {
        stream_.isTx = tx;
        stream_.channel = 0;
        stream_.fifoSize = fifo_samples;
        stream_.throughputVsLatency = 1.0F;
        stream_.dataFmt = static_cast<decltype(stream_.dataFmt)>(0);
        stream_.linkFmt = static_cast<decltype(stream_.linkFmt)>(2);
        check(LMS_SetupStream(device_, &stream_), "LMS_SetupStream");
        setup_ = true;
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    ~Stream() {
        if (started_) {
            LMS_StopStream(&stream_);
        }
        if (setup_) {
            LMS_DestroyStream(device_, &stream_);
        }
    }

    void start() {
        check(LMS_StartStream(&stream_), "LMS_StartStream");
        started_ = true;
    }

    void stop() {
        if (started_) {
            check(LMS_StopStream(&stream_), "LMS_StopStream");
            started_ = false;
        }
    }

    lms_stream_t* get() { return &stream_; }

private:
    lms_device_t* device_ = nullptr;
    lms_stream_t stream_{};
    bool setup_ = false;
    bool started_ = false;
};

Counts read_status(Stream& stream) {
    lms_stream_status_t status{};
    check(LMS_GetStreamStatus(stream.get(), &status), "LMS_GetStreamStatus");
    return {
        static_cast<std::uint64_t>(status.underrun),
        static_cast<std::uint64_t>(status.overrun),
        static_cast<std::uint64_t>(status.droppedPackets),
    };
}

void require_clean(const Counts& counts, const std::string& name) {
    if (counts.underrun != 0 || counts.overrun != 0 || counts.dropped != 0) {
        fail(name + " reported a stream error.");
    }
}

void configure_rx(lms_device_t* device, const Options& options) {
    check(LMS_EnableChannel(device, LMS_CH_RX, 0, true), "LMS_EnableChannel RX");
    check(LMS_SetLOFrequency(device, LMS_CH_RX, 0, options.frequency_hz),
          "LMS_SetLOFrequency RX");
    check(LMS_SetLPFBW(device, LMS_CH_RX, 0, options.bandwidth_hz), "LMS_SetLPFBW RX");
    check(LMS_SetGFIRLPF(device, LMS_CH_RX, 0, true, options.bandwidth_hz),
          "LMS_SetGFIRLPF RX");
    check(LMS_SetGaindB(device, LMS_CH_RX, 0, options.rx_gain_db), "LMS_SetGaindB RX");
    check(LMS_SetAntenna(device, LMS_CH_RX, 0, LMS_PATH_LNAW), "LMS_SetAntenna RX");
    check(LMS_Calibrate(device, LMS_CH_RX, 0, options.bandwidth_hz, 0), "LMS_Calibrate RX");
}

void configure_tx(lms_device_t* device, const Options& options) {
    check(LMS_EnableChannel(device, LMS_CH_TX, 0, true), "LMS_EnableChannel TX");
    check(LMS_SetLOFrequency(device, LMS_CH_TX, 0, options.frequency_hz),
          "LMS_SetLOFrequency TX");
    check(LMS_SetLPFBW(device, LMS_CH_TX, 0, options.bandwidth_hz), "LMS_SetLPFBW TX");
    check(LMS_SetGFIRLPF(device, LMS_CH_TX, 0, true, options.bandwidth_hz),
          "LMS_SetGFIRLPF TX");
    check(LMS_SetGaindB(device, LMS_CH_TX, 0, options.tx_gain_db), "LMS_SetGaindB TX");
    check(LMS_SetAntenna(device, LMS_CH_TX, 0, LMS_PATH_TX2), "LMS_SetAntenna TX");
    check(LMS_Calibrate(device, LMS_CH_TX, 0, options.bandwidth_hz, 0), "LMS_Calibrate TX");
    check(LMS_SetAntenna(device, LMS_CH_TX, 0, LMS_PATH_NONE), "LMS_SetAntenna TX off");
}

struct RxResult {
    std::uint64_t first_timestamp = 0;
    std::uint64_t next_timestamp = 0;
    std::uint64_t samples = 0;
};

int receive_chunk(Stream& stream, std::vector<std::complex<float>>& buffer,
                  lms_stream_meta_t& metadata) {
    const int count = LMS_RecvStream(
        stream.get(), buffer.data(), buffer.size(), &metadata, kTimeoutMs);
    if (count <= 0) {
        fail("LMS_RecvStream did not return samples.");
    }
    return count;
}

RxResult capture_off(Stream& stream, const Options& options) {
    std::ofstream output(options.tx_off_rx_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail("Could not open the transmitter-off capture.");
    }
    std::vector<std::complex<float>> buffer(kChunkSamples);
    std::uint64_t consumed = 0;
    std::uint64_t written = 0;
    std::uint64_t expected = 0;
    std::uint64_t first = 0;
    bool have_timestamp = false;

    while (consumed < options.capture_samples) {
        lms_stream_meta_t metadata{};
        const int count = receive_chunk(stream, buffer, metadata);
        if (have_timestamp && metadata.timestamp != expected) {
            fail("Transmitter-off RX timestamp discontinuity: expected " +
                 std::to_string(expected) + ", received " +
                 std::to_string(metadata.timestamp) + ".");
        }
        if (!have_timestamp) {
            first = metadata.timestamp;
            have_timestamp = true;
        }
        expected = metadata.timestamp + static_cast<std::uint64_t>(count);
        const auto keep = static_cast<std::size_t>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(count), options.capture_samples - consumed));
        output.write(reinterpret_cast<const char*>(buffer.data()),
                     static_cast<std::streamsize>(keep * sizeof(std::complex<float>)));
        written += keep;
        consumed += keep;
    }
    if (!output || written != options.capture_samples) {
        fail("Transmitter-off capture has the wrong size.");
    }
    return {first, first + written, written};
}

RxResult warm_up_rx(Stream& stream, std::uint64_t minimum_samples) {
    std::vector<std::complex<float>> buffer(kChunkSamples);
    std::uint64_t continuous = 0;
    std::uint64_t first = 0;
    std::uint64_t next = 0;
    bool have_timestamp = false;
    while (continuous < minimum_samples) {
        lms_stream_meta_t metadata{};
        const int count = receive_chunk(stream, buffer, metadata);
        if (!have_timestamp || metadata.timestamp != next) {
            first = metadata.timestamp;
            continuous = 0;
        }
        have_timestamp = true;
        next = metadata.timestamp + static_cast<std::uint64_t>(count);
        continuous += static_cast<std::uint64_t>(count);
    }
    return {first, next, continuous};
}

void send_session(Stream& stream, const std::vector<std::complex<float>>& samples,
                  std::uint64_t start_timestamp,
                  std::atomic<bool>& abort, std::exception_ptr& error) {
    try {
        std::uint64_t sent = 0;
        while (sent < samples.size() && !abort.load()) {
            const auto request = static_cast<std::size_t>(std::min<std::uint64_t>(
                kChunkSamples, samples.size() - sent));
            lms_stream_meta_t metadata{};
            metadata.timestamp = start_timestamp + sent;
            metadata.waitForTimestamp = true;
            metadata.flushPartialPacket = sent + request == samples.size();
            const int count = LMS_SendStream(
                stream.get(), samples.data() + sent, request, &metadata, kTimeoutMs);
            if (count <= 0 || count > static_cast<int>(request)) {
                fail("LMS_SendStream did not accept samples.");
            }
            sent += static_cast<std::uint64_t>(count);
        }
    } catch (...) {
        error = std::current_exception();
        abort.store(true);
    }
}

RxResult capture_session(Stream& stream, const Options& options,
                         std::uint64_t start_timestamp,
                         std::uint64_t initial_expected_timestamp,
                         const std::atomic<bool>& abort,
                         std::vector<std::complex<float>>& captured) {
    std::vector<std::complex<float>> buffer(kChunkSamples);
    std::uint64_t written = 0;
    std::uint64_t expected = initial_expected_timestamp;
    std::uint64_t first = 0;
    bool have_timestamp = false;

    while (written < options.session_samples) {
        if (abort.load()) {
            fail("Transmit stream stopped early.");
        }
        lms_stream_meta_t metadata{};
        const int count = receive_chunk(stream, buffer, metadata);
        if (abort.load()) {
            fail("Transmit stream stopped early.");
        }
        if (metadata.timestamp != expected) {
            fail("Session RX timestamp discontinuity: expected " +
                 std::to_string(expected) + ", received " +
                 std::to_string(metadata.timestamp) + ".");
        }
        if (!have_timestamp) {
            first = metadata.timestamp;
            have_timestamp = true;
        }
        expected = metadata.timestamp + static_cast<std::uint64_t>(count);
        const std::uint64_t chunk_start = metadata.timestamp;
        const std::uint64_t chunk_stop = expected;
        if (chunk_stop <= start_timestamp) {
            continue;
        }
        const std::uint64_t keep_start = std::max(chunk_start, start_timestamp);
        const std::uint64_t keep_stop = std::min(
            chunk_stop, start_timestamp + options.session_samples);
        if (keep_stop > keep_start) {
            const auto offset = static_cast<std::size_t>(keep_start - chunk_start);
            const auto keep = static_cast<std::size_t>(keep_stop - keep_start);
            std::copy_n(buffer.data() + offset, keep, captured.data() + written);
            written += keep;
        }
    }
    if (written != options.session_samples) {
        fail("Session capture has the wrong size.");
    }
    return {first, expected, written};
}

std::vector<std::complex<float>> read_session(const Options& options) {
    std::vector<std::complex<float>> samples(
        static_cast<std::size_t>(options.session_samples));
    std::ifstream input(options.tx_path, std::ios::binary);
    if (!input) {
        fail("Could not open the transmit file.");
    }
    input.read(reinterpret_cast<char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(samples.front())));
    if (!input || input.gcount() !=
        static_cast<std::streamsize>(samples.size() * sizeof(samples.front()))) {
        fail("Could not read the transmit file.");
    }
    return samples;
}

void write_session(const std::filesystem::path& path,
                   const std::vector<std::complex<float>>& samples) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail("Could not open the session capture.");
    }
    output.write(reinterpret_cast<const char*>(samples.data()),
                 static_cast<std::streamsize>(samples.size() * sizeof(samples.front())));
    if (!output) {
        fail("Could not write the session capture.");
    }
}

void preflight_output(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail("Could not prepare an output file.");
    }
}

void write_status(const Options& options, double rx_host_rate, double rx_rf_rate,
                  double tx_host_rate, double tx_rf_rate, double rx_lpf_bandwidth,
                  double tx_lpf_bandwidth, std::uint64_t start_timestamp,
                  const RxResult& off, const RxResult& session,
                  const Counts& off_counts, const Counts& rx_counts,
                  const Counts& tx_counts) {
    std::ofstream output(options.status_path, std::ios::trunc);
    if (!output) {
        fail("Could not open the status file.");
    }
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema_version\": \"sdr_stream_v1\",\n"
           << "  \"valid\": true,\n"
           << "  \"rx_host_rate_hz\": " << rx_host_rate << ",\n"
           << "  \"rx_rf_rate_hz\": " << rx_rf_rate << ",\n"
           << "  \"tx_host_rate_hz\": " << tx_host_rate << ",\n"
           << "  \"tx_rf_rate_hz\": " << tx_rf_rate << ",\n"
           << "  \"requested_filter_bandwidth_hz\": " << options.bandwidth_hz << ",\n"
           << "  \"rx_lpf_bandwidth_hz\": " << rx_lpf_bandwidth << ",\n"
           << "  \"tx_lpf_bandwidth_hz\": " << tx_lpf_bandwidth << ",\n"
           << "  \"gfir_enabled\": true,\n"
           << "  \"tx_start_timestamp\": " << start_timestamp << ",\n"
           << "  \"tx_off_first_timestamp\": " << off.first_timestamp << ",\n"
           << "  \"tx_off_next_timestamp\": " << off.next_timestamp << ",\n"
           << "  \"tx_off_samples\": " << off.samples << ",\n"
           << "  \"session_first_timestamp\": " << session.first_timestamp << ",\n"
           << "  \"session_next_timestamp\": " << session.next_timestamp << ",\n"
           << "  \"session_samples\": " << session.samples << ",\n"
           << "  \"tx_off_underrun\": " << off_counts.underrun << ",\n"
           << "  \"tx_off_overrun\": " << off_counts.overrun << ",\n"
           << "  \"tx_off_dropped_packets\": " << off_counts.dropped << ",\n"
           << "  \"rx_underrun\": " << rx_counts.underrun << ",\n"
           << "  \"rx_overrun\": " << rx_counts.overrun << ",\n"
           << "  \"rx_dropped_packets\": " << rx_counts.dropped << ",\n"
           << "  \"tx_underrun\": " << tx_counts.underrun << ",\n"
           << "  \"tx_overrun\": " << tx_counts.overrun << ",\n"
           << "  \"tx_dropped_packets\": " << tx_counts.dropped << "\n"
           << "}\n";
    if (!output) {
        fail("Could not write the status file.");
    }
}

int run(const Options& options) {
    static_assert(sizeof(std::complex<float>) == 8);
    if (options.session_samples >
        std::numeric_limits<std::size_t>::max() / sizeof(std::complex<float>)) {
        fail("Session size is too large.");
    }
    const auto expected_bytes = options.session_samples * sizeof(std::complex<float>);
    if (!std::filesystem::is_regular_file(options.tx_path) ||
        std::filesystem::file_size(options.tx_path) != expected_bytes) {
        fail("Transmit file has the wrong size.");
    }
    if (options.tx_gain_db > 73 || options.rx_gain_db > 73 ||
        options.fifo_samples < 1'048'576 || options.capture_samples == 0 ||
        options.session_samples == 0 || options.start_delay_samples == 0) {
        fail("Worker settings are outside their limits.");
    }
    preflight_output(options.session_rx_path);
    preflight_output(options.tx_off_rx_path);
    preflight_output(options.status_path);
    const auto tx_samples = read_session(options);
    std::vector<std::complex<float>> session_capture(
        static_cast<std::size_t>(options.session_samples));

    Device device(options.serial);
    check(LMS_SetSampleRate(device.get(), options.sample_rate_hz, 0), "LMS_SetSampleRate");
    configure_rx(device.get(), options);

    RxResult off_result;
    Counts off_counts;
    {
        Stream off_rx(device.get(), false, options.fifo_samples);
        off_rx.start();
        warm_up_rx(
            off_rx, std::max(options.settle_samples, options.start_delay_samples));
        read_status(off_rx);
        off_result = capture_off(off_rx, options);
        off_counts = read_status(off_rx);
        require_clean(off_counts, "Transmitter-off RX");
        off_rx.stop();
    }

    check(LMS_EnableChannel(device.get(), LMS_CH_TX, 0, true), "LMS_EnableChannel TX");
    check(LMS_SetSampleRate(device.get(), options.sample_rate_hz, 0), "LMS_SetSampleRate");
    configure_rx(device.get(), options);
    configure_tx(device.get(), options);
    double rx_host_rate = 0.0;
    double rx_rf_rate = 0.0;
    double tx_host_rate = 0.0;
    double tx_rf_rate = 0.0;
    double rx_lpf_bandwidth = 0.0;
    double tx_lpf_bandwidth = 0.0;
    check(LMS_GetSampleRate(device.get(), LMS_CH_RX, 0, &rx_host_rate, &rx_rf_rate),
          "LMS_GetSampleRate RX");
    check(LMS_GetSampleRate(device.get(), LMS_CH_TX, 0, &tx_host_rate, &tx_rf_rate),
          "LMS_GetSampleRate TX");
    check(LMS_GetLPFBW(device.get(), LMS_CH_RX, 0, &rx_lpf_bandwidth),
          "LMS_GetLPFBW RX");
    check(LMS_GetLPFBW(device.get(), LMS_CH_TX, 0, &tx_lpf_bandwidth),
          "LMS_GetLPFBW TX");

    Stream rx(device.get(), false, options.fifo_samples);
    Stream tx(device.get(), true, options.fifo_samples);
    rx.start();
    const RxResult warmup = warm_up_rx(
        rx, std::max(options.settle_samples, options.start_delay_samples));
    read_status(rx);
    const std::uint64_t rx_next = warmup.next_timestamp;
    const std::uint64_t start_timestamp = rx_next + options.start_delay_samples;
    check(LMS_SetAntenna(device.get(), LMS_CH_TX, 0, LMS_PATH_TX2),
          "LMS_SetAntenna TX active");
    tx.start();

    std::atomic<bool> abort{false};
    std::exception_ptr tx_error;
    std::thread tx_thread(
        send_session, std::ref(tx), std::cref(tx_samples), start_timestamp,
        std::ref(abort), std::ref(tx_error));
    RxResult session_result;
    try {
        session_result = capture_session(
            rx, options, start_timestamp, rx_next, abort, session_capture);
    } catch (...) {
        const auto rx_error = std::current_exception();
        abort.store(true);
        silence_tx(device.get());
        try {
            tx.stop();
        } catch (...) {
        }
        tx_thread.join();
        if (tx_error) {
            std::rethrow_exception(tx_error);
        }
        std::rethrow_exception(rx_error);
    }
    tx_thread.join();
    if (tx_error) {
        silence_tx(device.get());
        tx.stop();
        std::rethrow_exception(tx_error);
    }

    const Counts tx_counts = read_status(tx);
    tx.stop();
    check(LMS_SetAntenna(device.get(), LMS_CH_TX, 0, LMS_PATH_NONE),
          "LMS_SetAntenna TX off");
    check(LMS_SetGaindB(device.get(), LMS_CH_TX, 0, 0), "LMS_SetGaindB TX off");
    check(LMS_EnableChannel(device.get(), LMS_CH_TX, 0, false),
          "LMS_EnableChannel TX off");
    const Counts rx_counts = read_status(rx);
    rx.stop();
    require_clean(tx_counts, "Session TX");
    require_clean(rx_counts, "Session RX");
    write_session(options.session_rx_path, session_capture);

    write_status(options, rx_host_rate, rx_rf_rate, tx_host_rate, tx_rf_rate,
                 rx_lpf_bandwidth, tx_lpf_bandwidth, start_timestamp, off_result,
                 session_result, off_counts, rx_counts, tx_counts);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
