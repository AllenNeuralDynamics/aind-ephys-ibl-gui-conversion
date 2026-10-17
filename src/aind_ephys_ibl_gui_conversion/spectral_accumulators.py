"""Existing spectral reductions as independent window consumers."""

from __future__ import annotations

import numpy as np

from aind_ephys_ibl_gui_conversion.window_pipeline import Product, Spectrum


def parseval_rms(
    spectrum: np.ndarray, mask: np.ndarray, n_samples: int
) -> np.ndarray:
    """Preserve the existing one-sided, interior-band RMS calculation."""
    power = np.sum(np.abs(spectrum[mask]) ** 2, axis=0)
    return np.asarray(np.sqrt(2.0 * power / n_samples**2))


class RmsAccumulator:
    """Retain a per-window RMS time series from a shared spectrum."""

    def __init__(
        self,
        n_windows: int,
        n_channels: int,
        mask: np.ndarray,
        n_samples: int,
        correction: float = 1.0,
    ) -> None:
        """Allocate only the intended output and retain fixed band metadata."""
        self.values = np.zeros((n_windows, n_channels), dtype=np.float32)
        self.mask = mask
        self.n_samples = n_samples
        self.correction = correction

    def update(self, window_index: int, inputs: tuple[Product, ...]) -> None:
        """Write this window's RMS without modifying the FFT coefficients."""
        (spectrum,) = inputs
        assert isinstance(spectrum, Spectrum)
        self.values[window_index] = (
            parseval_rms(spectrum.coefficients, self.mask, self.n_samples)
            * self.correction
        )

    def finish(self) -> np.ndarray:
        """Return the RMS time series in microvolts."""
        return self.values


class PsdAccumulator:
    """Sum selected-bin power without retaining window spectra."""

    def __init__(self, mask: np.ndarray, n_channels: int) -> None:
        """Allocate the existing float64 PSD accumulator."""
        self.mask = mask
        self.power = np.zeros((int(mask.sum()), n_channels), dtype=np.float64)

    def update(self, window_index: int, inputs: tuple[Product, ...]) -> None:
        """Accumulate power from the borrowed spectrum."""
        (spectrum,) = inputs
        assert isinstance(spectrum, Spectrum)
        self.power += np.abs(spectrum.coefficients[self.mask]) ** 2

    def finish(self) -> np.ndarray:
        """Return unnormalized power for the existing normalization path."""
        return self.power


class CrossSpectralAccumulator:
    """Sum channel cross-products by band from the same shared FFT."""

    def __init__(self, masks: dict[str, np.ndarray], n_channels: int) -> None:
        """Allocate the existing complex128 per-band state."""
        self.masks = masks
        self.cross_power = {
            name: np.zeros((n_channels, n_channels), dtype=np.complex128)
            for name in masks
        }

    def update(self, window_index: int, inputs: tuple[Product, ...]) -> None:
        """Accumulate cross-power without mutating a shared product."""
        (spectrum,) = inputs
        assert isinstance(spectrum, Spectrum)
        for name, mask in self.masks.items():
            band = spectrum.coefficients[mask]
            self.cross_power[name] += band.conj().T @ band

    def finish(self) -> dict[str, np.ndarray]:
        """Return unnormalized cross-power for existing coherency outputs."""
        return self.cross_power
