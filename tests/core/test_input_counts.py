import h5py
import numpy as np

from pilot_proxy.input_counts import changes, load_good_inputs, read_good_inputs, save_good_inputs, span_counts

FRAME0, SEQ_NS, FRAME = 1_000_000_000_000, 5120, 8192


def write_vis(path, first_frame, mask, streams):
    """Write the bad_feed_mask group of one visibility file: mask [time, stream, pol, dish]."""
    seq = (first_frame + np.arange(len(mask))) * FRAME
    with h5py.File(path, "w") as handle:
        handle.attrs["frame0_unix_ns"] = FRAME0
        handle.attrs["fpga_seq_length_ns"] = SEQ_NS
        group = handle.create_group("bad_feed_mask")
        group["fpga_seq_num"] = seq.astype(np.uint64)
        group["mask"] = mask.astype(np.int8)
        group["stream_freq_id"] = np.array(streams, np.int32)
    return str(path)


def test_good_inputs_follow_the_mask_of_each_stream(tmp_path):
    streams = [[2531, 2547], [2715, 3115]]
    mask = np.ones((6, 2, 2, 4), np.int8)
    mask[:2, 0] = -1              # stream 0's mask frame had not reached the writer: dropped
    mask[4:, 0, 1, 3] = 0         # one input of stream 0 flagged from frame 4
    # the next file repeats the last two frames, which are kept once; the first file's copy of frame 5 is -1
    second = write_vis(tmp_path / "vis_2.h5", 4, mask[4:].repeat(2, axis=0)[:3], streams)
    mask[5, 0] = -1
    first = write_vis(tmp_path / "vis_1.h5", 0, mask, streams)
    (tmp_path / "vis_3.h5").write_bytes(b"not hdf5")      # a partial file is skipped
    # a stream with no mask frame in any file gives no counts
    write_vis(tmp_path / "vis_0.h5", 0, np.full((2, 1, 2, 4), -1), [[2869]])
    counts = read_good_inputs([str(tmp_path / "vis_*.h5")], [2531, 2869, 3115, 9999])
    assert sorted(counts) == [2531, 3115]
    times, good = counts[2531]
    assert list(good) == [8, 8, 7, 7, 7]
    assert list(times) == [FRAME0 + i * FRAME * SEQ_NS for i in range(2, 7)]
    assert list(counts[3115][1]) == [8] * 7
    assert changes(times, good) == [(FRAME0 + 4 * FRAME * SEQ_NS, 8, 7)]
    assert span_counts(times, good, times[1], times[3]) == (7, 7, 8)
    assert span_counts(times, good, times[-1] + 1, times[-1] + 2) is None
    save_good_inputs(counts, tmp_path / "inputs.npz")
    loaded = load_good_inputs(tmp_path / "inputs.npz")
    assert all(np.array_equal(loaded[f][i], counts[f][i]) for f in counts for i in (0, 1))
    assert first != second
