"""Immutable block metadata for registering sampled-window metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.fft

from aind_ephys_ibl_gui_conversion.channel_metadata import (
    normalized_channel_groups,
    normalized_contact_ids,
)
from aind_ephys_ibl_gui_conversion.window_pipeline import (
    FftRequest,
    RawRequest,
    Source,
)

if TYPE_CHECKING:
    import spikeinterface as si


def metadata_array(values: np.ndarray) -> np.ndarray:
    """Own a small read-only metadata copy, independent of recording state."""
    array = np.array(values, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SourceContext:
    """Sampling and native channel geometry for one already-planned read.

    Samples supplied to accumulators are scaled to microvolts. Coordinates
    are probe-local micrometres. No recording handle or trace I/O is exposed.
    """

    sampling_frequency: float
    read_samples: int
    window_starts: np.ndarray
    channel_ids: np.ndarray
    locations: np.ndarray
    contact_ids: np.ndarray
    shank_indices: np.ndarray

    @classmethod
    def from_recording(
        cls,
        recording: si.BaseRecording,
        read_samples: int,
        starts_seconds: np.ndarray,
    ) -> SourceContext:
        """Snapshot geometry and the scheduler's integer sample offsets."""
        fs = recording.get_sampling_frequency()
        last_start = recording.get_num_samples() - read_samples
        if last_start < 0:
            raise ValueError("Source is shorter than the planned read")
        starts = np.clip((starts_seconds * fs).astype(int), 0, last_start)
        return cls(
            sampling_frequency=float(fs),
            read_samples=read_samples,
            window_starts=metadata_array(starts),
            channel_ids=metadata_array(recording.get_channel_ids()),
            locations=metadata_array(recording.get_channel_locations()),
            contact_ids=metadata_array(normalized_contact_ids(recording)),
            shank_indices=metadata_array(normalized_channel_groups(recording)),
        )

    @property
    def n_channels(self) -> int:
        """Return the native channel count, without canonical-row sorting."""
        return len(self.channel_ids)


@dataclass(frozen=True)
class BlockContext:
    """Resolved scheduling and shared transform contracts for one block."""

    block_index: int
    block_source: Literal["main", "surface"]
    duration_seconds: float
    window_starts_seconds: np.ndarray
    timestamps: np.ndarray
    ap: SourceContext
    lfp: SourceContext
    ap_fft: FftRequest
    lfp_fft: FftRequest
    lfp_window_power: float
    block_id: str

    @property
    def n_windows(self) -> int:
        """Return the actual adaptive schedule length."""
        return len(self.timestamps)

    @property
    def separate_lfp(self) -> bool:
        """Distinguish independent NP1 sources from NP2 source aliases."""
        return self.ap is not self.lfp

    def frequencies(self, request: FftRequest) -> np.ndarray:
        """Build frequency coordinates using the resolved source rate."""
        source = self.ap if request.source == "ap" else self.lfp
        return np.asarray(
            scipy.fft.rfftfreq(
                request.n_samples, d=1.0 / source.sampling_frequency
            )
        )


@dataclass(frozen=True)
class SnippetPlan:
    """Independent snippets inside existing reads, in native samples."""

    request: RawRequest
    window_indices: tuple[int, ...]
    edge_samples: int
    sampling_frequency: float
    starts: tuple[int, ...]

    @property
    def retained_samples(self) -> int:
        """Return the analyzed core length after removing both filter edges."""
        return self.request.n_samples - 2 * self.edge_samples

    @property
    def seconds_per_window(self) -> float:
        """Return the core duration, without artifact-time subtraction."""
        return self.retained_samples / self.sampling_frequency

    def metadata(self) -> dict[str, object]:
        """Describe exact read and analyzed intervals in native samples."""
        return {
            "source": self.request.source,
            "sampling_frequency": self.sampling_frequency,
            "window_indices": self.window_indices,
            "sample_intervals": [
                [start, start + self.request.n_samples]
                for start in self.starts
            ],
            "analyzed_intervals": [
                [
                    start + self.edge_samples,
                    start + self.request.n_samples - self.edge_samples,
                ]
                for start in self.starts
            ],
            "analyzed_seconds": len(self.starts) * self.seconds_per_window,
        }


@dataclass(frozen=True)
class SampleBudget:
    """Select bounded, distributed snippets within the existing read schedule.

    Snippets are centered within their reads. A single snippet selects the
    middle scheduled window; larger budgets include the first and last.
    Updates may arrive out of time order and must be treated independently.
    """

    max_windows: int = 3
    seconds: float = 1.0
    edge_seconds: float = 0.025

    def resolve(
        self, source: SourceContext, name: Source = "ap"
    ) -> SnippetPlan:
        """Resolve the budget before reads and reject excess or overlap."""
        if not isinstance(self.max_windows, int) or self.max_windows < 1:
            raise ValueError("max_windows must be a positive integer")
        if not np.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Snippet seconds must be finite and positive")
        if not np.isfinite(self.edge_seconds) or self.edge_seconds < 0:
            raise ValueError("Filter edges must be finite and nonnegative")
        fs = source.sampling_frequency
        samples = round(self.seconds * fs)
        edge = round(self.edge_seconds * fs)
        if samples < 1 or samples > source.read_samples:
            raise ValueError("Snippet does not fit inside the existing read")
        if samples <= 2 * edge:
            raise ValueError("Snippet has no samples after filter edges")
        n_windows = len(source.window_starts)
        count = min(self.max_windows, n_windows)
        if count == 0:
            raise ValueError("No scheduled windows available")
        selected = (
            np.array([n_windows // 2])
            if count == 1
            else np.linspace(0, n_windows - 1, count).round().astype(int)
        )
        offset = (source.read_samples - samples) // 2
        starts = source.window_starts[selected] + offset
        if np.any(np.diff(starts) < samples - 2 * edge):
            raise ValueError("Selected analyzed intervals overlap")
        return SnippetPlan(
            RawRequest(name, samples, offset=offset),
            tuple(int(index) for index in selected),
            edge,
            fs,
            tuple(int(start) for start in starts),
        )
