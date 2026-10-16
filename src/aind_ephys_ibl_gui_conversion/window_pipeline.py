"""Block-local metric plans over shared, eagerly read sample windows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import numpy as np
import scipy.fft

Source = Literal["ap", "lfp"]


@dataclass(frozen=True)
class RawRequest:
    """Scaled samples in native channel order within an existing read."""

    source: Source
    n_samples: int
    offset: int = 0
    dtype: Literal["float32", "float64"] = "float32"


@dataclass(frozen=True)
class FftRequest:
    """An exact signal/transform specification, scoped to one block."""

    source: Source
    n_samples: int
    offset: int = 0
    reference: Literal["none", "median"] = "none"
    demean: bool = False
    taper: Literal["none", "hann"] = "none"
    dtype: Literal["float32", "float64"] = "float32"
    norm: Literal["backward", "ortho", "forward"] = "backward"


@dataclass(frozen=True)
class Spectrum:
    """FFT coefficients and the subtracted common-reference trace."""

    coefficients: np.ndarray
    reference: np.ndarray | None = None


Request = RawRequest | FftRequest
Product = np.ndarray | Spectrum


class Accumulator(Protocol):
    """Consume borrowed products and retain only owned result state."""

    def update(self, window_index: int, inputs: tuple[Product, ...]) -> None:
        """Reduce an independent window, possibly arriving out of order."""
        ...

    def finish(self) -> Any:
        """Return the metric's owned block result."""
        ...


@dataclass(frozen=True)
class MetricDefinition:
    """Declare read-only inputs and a block-local accumulator factory."""

    name: str
    inputs: tuple[Request, ...]
    create: Callable[[], Accumulator]
    window_indices: tuple[int, ...] | None = None


@dataclass
class ExecutionStats:
    """Deterministic work and product-lifetime counters for a block."""

    fft_calls: Counter[FftRequest] = field(default_factory=Counter)
    max_live_products: int = 0
    windows_processed: int = 0


@dataclass(frozen=True)
class _Step:
    """One prepared accumulator call and its product slots."""

    metric_index: int
    slots: tuple[int, ...]
    release: tuple[int, ...]


def _readonly(array: np.ndarray) -> np.ndarray:
    """Borrow an array without changing the owner's write permission."""
    view = array.view()
    view.flags.writeable = False
    return view


def _borrow(product: Product) -> Product:
    """Expose read-only views of a product without copying its buffers."""
    if isinstance(product, Spectrum):
        reference = product.reference
        return Spectrum(
            _readonly(product.coefficients),
            None if reference is None else _readonly(reference),
        )
    return _readonly(product)


class WindowPlan:
    """Compile metric calls once and release intermediates every window.

    This prototype supports full native channel order, raw slices and FFTs,
    and selections from an existing window schedule. Source sampling rates,
    scaling to microvolts, and geometry are fixed by the enclosing block.
    No recording, thread pool, or trace cache is held here.
    """

    def __init__(
        self,
        definitions: tuple[MetricDefinition, ...],
        *,
        n_windows: int,
        source_samples: dict[Source, int],
        separate_lfp: bool,
    ) -> None:
        """Validate requests and prepare schedules before allocating state."""
        self.definitions = definitions
        self.stats = ExecutionStats()
        self._requests: list[Request] = []
        metric_slots, selections = self._prepare_inputs(
            n_windows, source_samples, separate_lfp
        )
        order = self._execution_order(metric_slots)
        schedules: dict[tuple[int, ...], tuple[_Step, ...]] = {}
        self._schedules = []
        for window in range(n_windows):
            selected = tuple(i for i in order if window in selections[i])
            if selected not in schedules:
                schedules[selected] = self._make_schedule(
                    selected, metric_slots
                )
            self._schedules.append(schedules[selected])
        self._hann = {
            request: np.hanning(request.n_samples).astype(request.dtype)
            for request in self._requests
            if isinstance(request, FftRequest) and request.taper == "hann"
        }
        self._accumulators = tuple(d.create() for d in definitions)

    def _prepare_inputs(
        self,
        n_windows: int,
        source_samples: dict[Source, int],
        separate_lfp: bool,
    ) -> tuple[list[tuple[int, ...]], list[set[int]]]:
        """Resolve exact input slots and validate selections once per block."""
        slots: dict[Request, int] = {}
        metric_slots = []
        selections = []
        if len({d.name for d in self.definitions}) != len(self.definitions):
            raise ValueError("Metric names must be unique")
        for definition in self.definitions:
            selected = (
                tuple(range(n_windows))
                if definition.window_indices is None
                else definition.window_indices
            )
            if len(set(selected)) != len(selected) or any(
                i < 0 or i >= n_windows for i in selected
            ):
                raise ValueError("Invalid or duplicate selected window index")
            selections.append(set(selected))
            input_slots = []
            for request in definition.inputs:
                self._validate_request(request, source_samples)
                if request.source == "lfp" and not separate_lfp:
                    request = replace(request, source="ap")
                if request not in slots:
                    slots[request] = len(self._requests)
                    self._requests.append(request)
                input_slots.append(slots[request])
            metric_slots.append(tuple(input_slots))
        return metric_slots, selections

    def _execution_order(
        self, metric_slots: list[tuple[int, ...]]
    ) -> list[int]:
        """Run raw reducers before allocating FFTs, preserving stable order."""
        return sorted(
            range(len(self.definitions)),
            key=lambda index: any(
                isinstance(self._requests[slot], FftRequest)
                for slot in metric_slots[index]
            ),
        )

    @staticmethod
    def _make_schedule(
        order: tuple[int, ...], metric_slots: list[tuple[int, ...]]
    ) -> tuple[_Step, ...]:
        """Precompute the last use of every product in a selected schedule."""
        remaining = Counter(slot for i in order for slot in metric_slots[i])
        steps = []
        for i in order:
            released = []
            for slot in metric_slots[i]:
                remaining[slot] -= 1
                if remaining[slot] == 0:
                    released.append(slot)
            steps.append(_Step(i, metric_slots[i], tuple(released)))
        return tuple(steps)

    def _validate_request(
        self, request: Request, source_samples: dict[Source, int]
    ) -> None:
        """Validate every requested product before any accumulator is made."""
        if request.source not in source_samples:
            raise ValueError(f"Unknown source: {request.source}")
        if request.offset < 0 or request.n_samples < 1:
            raise ValueError("Sample interval must be positive")
        if request.offset + request.n_samples > source_samples[request.source]:
            raise ValueError("Metric requires samples outside the read")
        if request.dtype not in {"float32", "float64"}:
            raise ValueError("Unsupported sample dtype")
        if isinstance(request, FftRequest):
            self._validate_fft(request)

    @staticmethod
    def _validate_fft(request: FftRequest) -> None:
        """Reject unsupported semantics rather than sharing approximately."""
        if request.reference not in {"none", "median"}:
            raise ValueError("Unsupported reference")
        if request.taper not in {"none", "hann"}:
            raise ValueError("Unsupported taper")
        if request.norm not in {"backward", "ortho", "forward"}:
            raise ValueError("Unsupported FFT normalization")

    def _produce(
        self, request: Request, sources: dict[Source, np.ndarray]
    ) -> Product:
        """Materialize one exact product from already-read raw buffers."""
        stop = request.offset + request.n_samples
        raw = sources[request.source][request.offset : stop]
        if raw.shape[0] != request.n_samples:
            raise ValueError("Read returned fewer samples than planned")
        if isinstance(request, RawRequest):
            return raw.astype(request.dtype, copy=False)
        data = raw.astype(request.dtype)
        reference = None
        if request.reference == "median":
            reference = np.median(data, axis=1, keepdims=True)
            np.subtract(data, reference, out=data)
        if request.demean:
            np.subtract(data, data.mean(axis=0), out=data)
        if request.taper == "hann":
            np.multiply(data, self._hann[request][:, None], out=data)
        coefficients = scipy.fft.rfft(data, axis=0, norm=request.norm)
        self.stats.fft_calls[request] += 1
        return Spectrum(
            coefficients, None if reference is None else reference[:, 0]
        )

    def consume(
        self, window_index: int, ap: np.ndarray, lfp: np.ndarray | None
    ) -> None:
        """Update selected metrics and discard all borrowed products."""
        sources: dict[Source, np.ndarray] = {
            "ap": ap,
            "lfp": ap if lfp is None else lfp,
        }
        products: list[Product | None] = [None] * len(self._requests)
        live = 0
        for step in self._schedules[window_index]:
            inputs = []
            for slot in step.slots:
                product = products[slot]
                if product is None:
                    product = self._produce(self._requests[slot], sources)
                    products[slot] = product
                    live += 1
                    self.stats.max_live_products = max(
                        self.stats.max_live_products, live
                    )
                inputs.append(_borrow(product))
            self._accumulators[step.metric_index].update(
                window_index, tuple(inputs)
            )
            # Loop locals must not retain a released FFT while the next
            # provider allocates its output.
            inputs.clear()
            if step.slots:
                del product
            for slot in step.release:
                products[slot] = None
                live -= 1
        self.stats.windows_processed += 1

    def finish(self) -> dict[str, Any]:
        """Return named block results without retaining sample products."""
        return {
            definition.name: accumulator.finish()
            for definition, accumulator in zip(
                self.definitions, self._accumulators, strict=True
            )
        }
