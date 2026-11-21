"""recv files in the layout of kotekan#1925 (record_per_frame_metadata), built from one-frame recv files."""
import h5py
import numpy as np


def write_frames_file(sources, path, per_frame_time=False):
    """Write the frames of one-frame recv files, in the given (arrival) order, as one several-frame file.

    As in Kotekan, the dataset attributes are the first frame's, the frames are stacked along the first axis, and
    frame_metadata holds each frame's FPGA_SEQ_NUM, COARSE_FREQ, TIME_DOWNSAMPLING_FPGA and dim. Kotekan starts a
    new file when a frame's keys differ, so every source must have the same ones. per_frame_time adds
    FPGA_SEQ_TIME_NSEC, which Kotekan does not write.
    """
    blocks, attrs = [], []
    for source in sources:
        with h5py.File(source, "r") as handle:
            name = next(n for n in handle if n.startswith("dtv_"))
            blocks.append(handle[name][()])
            attrs.append(dict(handle[name].attrs))
    keys = {key for key in ("coarse_freq", "time_downsampling_fpga") if key in attrs[0]}
    if any({key for key in ("coarse_freq", "time_downsampling_fpga") if key in a} != keys for a in attrs):
        raise ValueError("frames with other metadata keys go to another file")
    with h5py.File(path, "w") as out:
        data = out.create_dataset(name, data=np.concatenate(blocks), chunks=blocks[0].shape)
        for key, value in attrs[0].items():
            data.attrs[key] = value
        meta = out.create_group("frame_metadata")
        meta["FPGA_SEQ_NUM"] = np.array([a["fpga_seq_num"] for a in attrs], np.uint64)
        if "coarse_freq" in keys:
            meta["COARSE_FREQ"] = np.array([a["coarse_freq"] for a in attrs], np.int64)
        if "time_downsampling_fpga" in keys:
            meta["TIME_DOWNSAMPLING_FPGA"] = np.array([a["time_downsampling_fpga"] for a in attrs], np.int64)
        if per_frame_time:
            meta["FPGA_SEQ_TIME_NSEC"] = np.array([a["fpga_seq_time_nsec"] for a in attrs], np.int64)
        meta["dim"] = np.array([b.shape for b in blocks], np.int64)
    return path
