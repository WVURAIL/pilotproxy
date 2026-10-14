# coding=utf-8
"""Frame-chunk helpers for segmented CHIME inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .hdf5_input import ChimePilotDataset


@dataclass(frozen=True)
class ChimeFrameChunk:
    start_frame: int
    frames_in_chunk: int
    start_sample: int
    stop_sample: int


def available_frames(
    dataset: ChimePilotDataset,
    *,
    frame_size_samples: int,
) -> int:
    frame = int(frame_size_samples)
    if frame <= 0:
        raise ValueError("frame_size_samples must be positive.")
    return int(dataset.total_time_samples // frame)


def iter_frame_chunks(
    dataset: ChimePilotDataset,
    *,
    frame_size_samples: int,
    frames_per_chunk: int,
    max_frames: int | None = None,
) -> Iterator[ChimeFrameChunk]:
    """Yield contiguous frame chunks in logical segment-concatenated order."""
    frame = int(frame_size_samples)
    chunk_frames = int(frames_per_chunk)
    if frame <= 0:
        raise ValueError("frame_size_samples must be positive.")
    if chunk_frames <= 0:
        raise ValueError("frames_per_chunk must be positive.")

    total_frames = available_frames(dataset, frame_size_samples=frame)
    if max_frames is not None:
        total_frames = min(total_frames, int(max_frames))
    frame_index = 0
    while frame_index < total_frames:
        frames = min(chunk_frames, total_frames - frame_index)
        start_sample = frame_index * frame
        stop_sample = start_sample + frames * frame
        yield ChimeFrameChunk(
            start_frame=int(frame_index),
            frames_in_chunk=int(frames),
            start_sample=int(start_sample),
            stop_sample=int(stop_sample),
        )
        frame_index += frames


__all__ = ["ChimeFrameChunk", "available_frames", "iter_frame_chunks"]
