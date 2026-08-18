"""Data types for ephys metric computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spikeinterface as si


@dataclass
class ExperimentBlock:
    """One experiment block within a probe stream."""

    recording: si.BaseRecording
    lfp_recording: si.BaseRecording | None  # 1.0 probes only
    block_index: int

    @property
    def is_1_0(self) -> bool:
        """True if this is a Neuropixels 1.0 probe (separate LFP)."""
        return self.lfp_recording is not None

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return (
            self.recording.get_num_samples()
            / self.recording.get_sampling_frequency()
        )


@dataclass
class ProbeStream:
    """All experiment blocks for one probe stream."""

    stream_name: str
    probe_name: str
    blocks: list[ExperimentBlock]
    output_folder: Path

    @property
    def main_block(self) -> ExperimentBlock:
        """The longest (main session) block."""
        return max(self.blocks, key=lambda b: b.duration)

    @property
    def has_surface(self) -> bool:
        """True if surface-finding blocks are present."""
        return len(self.blocks) > 1

    @property
    def combined_max_duration(self) -> float:
        """Shortest block duration (for combined RMS truncation)."""
        return min(b.duration for b in self.blocks)


@dataclass
class ChannelTable:
    """Per-probe channel metadata in canonical output order."""

    raw_ind: np.ndarray  # legacy positional rawInd, unchanged for now
    contact_id: np.ndarray  # physical probe contact ids in row order
    local_coordinates: np.ndarray  # (n_channels, 2)
    shank_ind: np.ndarray  # 0-based normalized shank index per channel


@dataclass
class ShankChannels:
    """Per-shank channel metadata from one block (depth-sorted)."""

    shank_index: int  # 1-based legacy file suffix; shankInd is 0-based
    channel_indices: np.ndarray  # block channel indices, depth-sorted
    locations: np.ndarray  # (n_shank_channels, 2), depth-sorted


@dataclass
class BlockMetrics:
    """All metrics computed for one experiment block."""

    block: ExperimentBlock
    rms_ap: np.ndarray  # (n_windows, n_channels)
    rms_lfp: np.ndarray  # (n_windows, n_channels)
    timestamps: np.ndarray  # (n_windows,)
    correlation: dict  # band -> full real coherency matrix
    coherency: dict  # band -> full complex coherency matrix
    psd_power: np.ndarray  # (n_lfp_freqs, n_channels)
    psd_freqs: np.ndarray  # (n_lfp_freqs,)
    shank_channels: list[ShankChannels]  # per-shank channel info
