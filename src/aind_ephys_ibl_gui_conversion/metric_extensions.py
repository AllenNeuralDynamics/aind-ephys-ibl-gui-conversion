"""Small registration, pure reduction, and output helpers for metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from aind_ephys_ibl_gui_conversion.channel_metadata import _channel_key
from aind_ephys_ibl_gui_conversion.metric_context import (
    BlockContext,
    SnippetPlan,
    SourceContext,
)
from aind_ephys_ibl_gui_conversion.types import ChannelTable
from aind_ephys_ibl_gui_conversion.window_pipeline import (
    FftRequest,
    MetricDefinition,
    Product,
    RawRequest,
    Request,
    Source,
    Spectrum,
)

T = TypeVar("T")


@dataclass(frozen=True)
class MetricBlockResult(Generic[T]):
    """An owned result paired with resolved block identity and provenance."""

    context: BlockContext
    value: T


@dataclass(frozen=True)
class StreamContext:
    """Canonical channel geometry for pure stream reduction, without I/O."""

    channel_table: ChannelTable

    def channel_rows(self, source: SourceContext) -> np.ndarray:
        """Map native contacts onto existing rows without interpolation."""
        by_key = {
            _channel_key(location): row
            for row, location in enumerate(
                self.channel_table.local_coordinates
            )
        }
        try:
            rows = np.array(
                [
                    by_key[_channel_key(location)]
                    for location in source.locations
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "Metric contacts are absent from channel table"
            ) from exc
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Metric contacts map to duplicate channel rows")
        return rows


class WindowMetric(Protocol):
    """Shared configuration with per-block state and explicit output names."""

    @property
    def name(self) -> str:
        """Return the configured instance name and result key."""
        ...

    @property
    def output_files(self) -> tuple[str, ...]:
        """Declare output paths relative to the stream folder before reads."""
        ...

    def define(self, context: BlockContext) -> MetricDefinition:
        """Declare independent windows and create fresh accumulator state."""
        ...

    def reduce(
        self,
        context: StreamContext,
        results: tuple[MetricBlockResult[Any], ...],
    ) -> Any:
        """Combine owned results in block order without writing files."""
        ...

    def write(self, output_folder: Path, result: Any) -> None:
        """Write the reduced result using declared output filenames."""
        ...


@dataclass(frozen=True)
class Metric:
    """Compose preparation, pure reduction, and saving from callables.

    Functions stored on this instance satisfy ``WindowMetric`` directly.
    Accumulation state belongs to the factory returned by ``define``.
    """

    name: str
    define: Callable[[BlockContext], MetricDefinition]
    reduce: Callable[[StreamContext, tuple[MetricBlockResult[Any], ...]], Any]
    write: Callable[[Path, Any], None]
    output_files: tuple[str, ...]


def validate_metrics(metrics: tuple[WindowMetric, ...]) -> None:
    """Reject duplicate identities and unsafe/colliding files before reads.

    Extensions use their own ALF/QC objects or the channels object. Existing
    channel metadata attributes and the existing ephys objects are reserved.
    """
    names = {"rms_ap", "rms_lfp", "psd", "cross_power"}
    files: set[str] = set()
    reserved_channels = {
        "localCoordinates",
        "rawInd",
        "contactId",
        "contactIds",
        "shankInd",
    }
    for metric in metrics:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", metric.name):
            raise ValueError("Metric name must be a simple identifier")
        if metric.name in names:
            raise ValueError("Metric names must be unique")
        names.add(metric.name)
        for filename in metric.output_files:
            if Path(filename).name != filename or filename in {"", ".", ".."}:
                raise ValueError("Metric outputs must be simple filenames")
            if filename in files:
                raise ValueError(f"Metric output collision: {filename}")
            parts = filename.split(".")
            if (
                parts[0].startswith(
                    (
                        "_iblqc_ephysTimeRmsAP",
                        "_iblqc_ephysTimeRmsLF",
                        "_iblqc_ephysSpectralDensityLF",
                        "_iblqc_metrics",
                    )
                )
                or filename == "band_corr"
                or (
                    parts[0] in {"channels", "_iblqc_channels"}
                    and len(parts) > 1
                    and parts[1] in reserved_channels
                )
            ):
                raise ValueError(f"Metric output is reserved: {filename}")
            files.add(filename)


@dataclass(frozen=True)
class ChannelOutput:
    """Scalar values and coverage in canonical channel-table order."""

    values: np.ndarray
    sampled_seconds: np.ndarray
    units: str
    metadata: dict[str, Any]

    def validate(self, n_rows: int) -> None:
        """Check shape and coverage before any output files are written."""
        if self.values.shape != (n_rows,) or self.sampled_seconds.shape != (
            n_rows,
        ):
            raise ValueError(
                "Channel result shape does not match channel table"
            )
        if np.any(~np.isfinite(self.sampled_seconds)) or np.any(
            self.sampled_seconds < 0
        ):
            raise ValueError("Sampled duration must be finite and nonnegative")
        if np.any(np.isinf(self.values)):
            raise ValueError("Channel values must be finite or missing (NaN)")
        if np.any((self.sampled_seconds == 0) & np.isfinite(self.values)):
            raise ValueError("Unmeasured channels must remain NaN")


def block_provenance(context: BlockContext, source: str) -> dict[str, Any]:
    """Describe the actual integer read plan once for channel writers."""
    info = context.ap if source == "ap" else context.lfp
    return {
        "block_id": context.block_id,
        "block_index": context.block_index,
        "source": context.block_source,
        "sampling_frequency": info.sampling_frequency,
        "read_start_samples": info.window_starts.tolist(),
        "read_samples": info.read_samples,
    }


def _channel_files(attribute: str) -> tuple[str, ...]:
    """Declare the standard scalar contact output filenames."""
    return (
        f"_iblqc_channels.{attribute}.npy",
        f"_iblqc_channels.{attribute}SampledDuration.npy",
        f"_iblqc_{attribute}.metadata.json",
    )


def _write_channel_output(
    output_folder: Path,
    result: ChannelOutput,
    *,
    name: str,
    attribute: str,
) -> None:
    """Save contact values, analyzed duration, and JSON method metadata."""
    result.validate(len(result.values))
    value_file, duration_file, metadata_file = _channel_files(attribute)
    metadata = {
        **result.metadata,
        "instance": name,
        "units": result.units,
        "sampled_duration_units": "s",
        "row_order": "channels.localCoordinates",
    }
    encoded = json.dumps(metadata, indent=2) + "\n"
    np.save(output_folder / value_file, result.values)
    np.save(output_folder / duration_file, result.sampled_seconds)
    (output_folder / metadata_file).write_text(encoded)


def channel_metric(
    *,
    name: str,
    define: Callable[[BlockContext], MetricDefinition],
    reduce: Callable[
        [StreamContext, tuple[MetricBlockResult[Any], ...]], ChannelOutput
    ],
    attribute: str | None = None,
) -> Metric:
    """Compose any contact-scalar estimator with the standard ALF writer."""
    attribute = name if attribute is None else attribute
    return Metric(
        name=name,
        define=define,
        reduce=reduce,
        write=partial(_write_channel_output, name=name, attribute=attribute),
        output_files=_channel_files(attribute),
    )


@dataclass(frozen=True)
class ChannelComputation:
    """Declare one input and its per-window channel computation.

    The mean helper derives geometry, analyzed duration, and input provenance.
    A snippet plan supplies selection and filter-edge lengths; the computation
    must filter and trim those edges before returning its channel values.
    """

    input: Request | SnippetPlan
    compute: Callable[..., np.ndarray]
    window_indices: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelMeanResult:
    """Duration-weighted channel sums and valid sampled seconds."""

    weighted_sum: np.ndarray
    sampled_seconds: np.ndarray
    input_source: Source | None = None
    provenance: dict[str, Any] | None = None


class ChannelMeanAccumulator:
    """Time-weight the channel values returned by a computation function.

    ``compute(*inputs)`` receives borrowed products in request order and
    returns one scalar per native channel. Nonfinite values exclude channel
    windows. Each block gets fresh state through its accumulator factory.
    """

    def __init__(
        self,
        n_channels: int,
        seconds: float,
        compute: Callable[..., np.ndarray],
        *,
        input_source: Source | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Bind the calculation and allocate only per-channel totals."""
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Window seconds must be finite and positive")
        self.seconds = seconds
        self.compute = compute
        self.input_source = input_source
        self.provenance = provenance
        self.weighted_sum = np.zeros(n_channels, dtype=np.float64)
        self.exposure = np.zeros(n_channels, dtype=np.float64)

    def update(self, window_index: int, inputs: tuple[Product, ...]) -> None:
        """Call the computation once and accumulate finite channel values."""
        values = np.asarray(self.compute(*inputs))
        if values.shape != self.weighted_sum.shape:
            raise ValueError("Window result shape must match native channels")
        valid = np.isfinite(values)
        self.weighted_sum[valid] += values[valid] * self.seconds
        self.exposure[valid] += self.seconds

    def finish(self) -> ChannelMeanResult:
        """Return owned sufficient statistics for time-weighted reduction."""
        return ChannelMeanResult(
            self.weighted_sum,
            self.exposure,
            self.input_source,
            self.provenance,
        )


def reduce_channel_mean(
    context: StreamContext,
    results: tuple[MetricBlockResult[ChannelMeanResult], ...],
    *,
    units: str,
    source: Source | None = None,
    output_scale: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> ChannelOutput:
    """Pool native totals by contact, weighting by valid sampled seconds.

    This pure reduction is for additive means. Robust summaries, sample-level
    rejection, and other estimators can supply a different reduction function.
    """
    n_rows = len(context.channel_table.raw_ind)
    total = np.zeros(n_rows, dtype=np.float64)
    exposure = np.zeros(n_rows, dtype=np.float64)
    provenance = []
    sources: set[Source] = set()
    if source is not None and source not in {"ap", "lfp"}:
        raise ValueError("Mean source must be ap or lfp")
    for result in results:
        value = result.value
        resolved_source = value.input_source or source
        if resolved_source not in {"ap", "lfp"}:
            raise ValueError("Channel mean requires its input source")
        if source is not None and source != resolved_source:
            raise ValueError("Mean source does not match prepared input")
        sources.add(resolved_source)
        info = (
            result.context.ap
            if resolved_source == "ap"
            else result.context.lfp
        )
        rows = context.channel_rows(info)
        if (
            value.weighted_sum.shape != rows.shape
            or value.sampled_seconds.shape != rows.shape
        ):
            raise ValueError("Channel mean shape does not match source")
        if np.any(~np.isfinite(value.weighted_sum)) or np.any(
            ~np.isfinite(value.sampled_seconds) | (value.sampled_seconds < 0)
        ):
            raise ValueError("Channel mean sums and coverage are invalid")
        if np.any((value.sampled_seconds == 0) & (value.weighted_sum != 0)):
            raise ValueError("Unmeasured channel sums must be zero")
        np.add.at(total, rows, value.weighted_sum)
        np.add.at(exposure, rows, value.sampled_seconds)
        provenance.append(
            {
                **(value.provenance or {}),
                **block_provenance(result.context, resolved_source),
                "input_source": resolved_source,
            }
        )
    mean = np.full(n_rows, np.nan, dtype=np.float64)
    np.divide(total, exposure, out=mean, where=exposure > 0)
    output = ChannelOutput(
        mean * output_scale,
        exposure,
        units,
        {
            **(metadata or {}),
            "input_source": (
                next(iter(sources)) if len(sources) == 1 else source
            ),
            "blocks": provenance,
            "pooling": "sampled-time weighted mean per channel-table row",
        },
    )
    output.validate(n_rows)
    return output


def _mean_window_indices(
    context: BlockContext,
    computation: ChannelComputation,
    snippet: SnippetPlan | None,
) -> tuple[int, ...]:
    """Resolve one selection and reject ambiguous or invalid window indices."""
    if snippet is not None:
        if computation.window_indices is not None:
            raise ValueError("Snippet plan already selects its windows")
        selected = snippet.window_indices
    else:
        selected = (
            tuple(range(context.n_windows))
            if computation.window_indices is None
            else computation.window_indices
        )
    if len(set(selected)) != len(selected) or any(
        not isinstance(index, (int, np.integer))
        or index < 0
        or index >= context.n_windows
        for index in selected
    ):
        raise ValueError("Invalid or duplicate selected window index")
    return tuple(int(index) for index in selected)


def _mean_sampling(
    context: BlockContext, computation: ChannelComputation
) -> tuple[Request, SourceContext, SnippetPlan]:
    """Derive geometry and exact analyzed intervals from the declared input."""
    snippet = (
        computation.input
        if isinstance(computation.input, SnippetPlan)
        else None
    )
    request = snippet.request if snippet is not None else computation.input
    if not isinstance(request, (RawRequest, FftRequest)):
        raise TypeError("Computation input must be a request or snippet plan")
    if request.source not in {"ap", "lfp"}:
        raise ValueError("Mean source must be ap or lfp")
    info = context.ap if request.source == "ap" else context.lfp
    if (
        request.offset < 0
        or request.n_samples < 1
        or (request.offset + request.n_samples > info.read_samples)
    ):
        raise ValueError("Computation input exceeds the existing read")
    selected = _mean_window_indices(context, computation, snippet)
    edge = 0 if snippet is None else snippet.edge_samples
    if edge < 0 or 2 * edge >= request.n_samples:
        raise ValueError("Computation has no valid analyzed core")
    sampling = SnippetPlan(
        request=RawRequest(request.source, request.n_samples, request.offset),
        window_indices=selected,
        edge_samples=edge,
        sampling_frequency=info.sampling_frequency,
        starts=tuple(
            int(info.window_starts[index]) + request.offset
            for index in selected
        ),
    )
    if snippet is not None and (
        snippet.sampling_frequency != sampling.sampling_frequency
        or snippet.starts != sampling.starts
    ):
        raise ValueError("Snippet plan does not match this block's source")
    return request, info, sampling


def _mean_definition(
    context: BlockContext,
    *,
    name: str,
    prepare: Callable[[BlockContext], ChannelComputation],
) -> MetricDefinition:
    """Bind the declared calculation to derived accumulation and provenance."""
    computation = prepare(context)
    request, info, sampling = _mean_sampling(context, computation)
    provenance = sampling.metadata()
    provenance.pop("source")
    return MetricDefinition(
        name=name,
        inputs=(request,),
        create=partial(
            ChannelMeanAccumulator,
            n_channels=info.n_channels,
            seconds=sampling.seconds_per_window,
            compute=computation.compute,
            input_source=request.source,
            provenance={
                **computation.metadata,
                **provenance,
                "request": asdict(request),
            },
        ),
        window_indices=sampling.window_indices,
    )


def _raw_computation(
    context: BlockContext,
    *,
    source: Source,
    compute: Callable[..., np.ndarray],
    window_indices: tuple[int, ...] | None,
) -> ChannelComputation:
    """Declare a raw calculation over the existing source read interval."""
    info = context.ap if source == "ap" else context.lfp
    return ChannelComputation(
        RawRequest(source, info.read_samples),
        compute,
        window_indices=window_indices,
    )


def channel_mean(
    *,
    name: str,
    units: str,
    source: Source | None = None,
    compute: Callable[[np.ndarray], np.ndarray] | None = None,
    prepare: Callable[[BlockContext], ChannelComputation] | None = None,
    window_indices: tuple[int, ...] | None = None,
    output_scale: float = 1.0,
    metadata: dict[str, Any] | None = None,
    attribute: str | None = None,
) -> Metric:
    """Build a contact mean from a raw function or per-block computation.

    Provide ``source`` and ``compute(traces)`` for the raw case. Alternatively,
    ``prepare(context)`` returns a ``ChannelComputation`` containing a request
    (or snippet plan) and a bound function. Geometry, analyzed duration, window
    selection, and exact input provenance are then derived from that input.
    """
    if not np.isfinite(output_scale):
        raise ValueError("Output scale must be finite")
    if prepare is not None:
        if (
            source is not None
            or compute is not None
            or window_indices is not None
        ):
            raise ValueError("Use preparation or raw source/compute arguments")
    else:
        if source is None or source not in {"ap", "lfp"} or compute is None:
            raise ValueError("Raw mean requires an ap/lfp source and compute")
        prepare = partial(
            _raw_computation,
            source=source,
            compute=compute,
            window_indices=window_indices,
        )
    return channel_metric(
        name=name,
        attribute=attribute,
        define=partial(_mean_definition, name=name, prepare=prepare),
        reduce=partial(
            reduce_channel_mean,
            source=source,
            units=units,
            output_scale=output_scale,
            metadata=dict(metadata or {}),
        ),
    )


def band_power(
    spectrum: Spectrum, *, mask: np.ndarray, scale: float
) -> np.ndarray:
    """Compute interior-band power from borrowed FFT coefficients."""
    power = np.sum(np.abs(spectrum.coefficients[mask]) ** 2, axis=0)
    return np.asarray(power * scale)


def _prepare_band_power(
    context: BlockContext,
    *,
    low_hz: float,
    high_hz: float,
    window_indices: tuple[int, ...] | None,
) -> ChannelComputation:
    """Resolve the scientific band calculation from the existing LF FFT."""
    request = context.lfp_fft
    fs = context.lfp.sampling_frequency
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError("Band must be strictly between DC and Nyquist")
    frequencies = context.frequencies(request)
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not mask.any():
        raise ValueError("Band contains no FFT bins")
    return ChannelComputation(
        input=request,
        compute=partial(
            band_power,
            mask=mask,
            scale=2.0 / (request.n_samples * context.lfp_window_power),
        ),
        window_indices=window_indices,
    )


def lfp_band_power(
    low_hz: float = 30.0,
    high_hz: float = 100.0,
    window_indices: tuple[int, ...] | None = None,
    name: str = "lfp_band_power",
) -> Metric:
    """Compose LF power with automatic sampling, geometry, and provenance."""
    return channel_mean(
        name=name,
        attribute="lfpBandPower" if name == "lfp_band_power" else name,
        prepare=partial(
            _prepare_band_power,
            low_hz=low_hz,
            high_hz=high_hz,
            window_indices=window_indices,
        ),
        units="V^2",
        output_scale=1e-12,
        metadata={
            "band_hz": [low_hz, high_hz],
            "high_edge_inclusive": False,
        },
    )
