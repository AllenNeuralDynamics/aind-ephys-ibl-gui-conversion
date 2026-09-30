"""Narrowband interference lines and the brain-surface depth they imply.

A contact above the brain surface couples to rig interference far more
strongly than one in tissue, and the transition is sharp, so the depth profile
of a line's amplitude locates the surface independently of histology.

Lines are found as peaks of spectral excess, each channel's power over a ring
of background around it, in an unreferenced Hann spectrum. Not a harmonic
F-test: the lines that discriminate the surface are often narrowband noise
whose frequency wanders by hertz, not stable sinusoids. The spectrum must be
unreferenced because the common-median reference cancels exactly these lines.

Ported from ``aind-ephys-noise-survey`` (``survey._line_profiles`` and
``channels``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.fft
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.special import expit


@dataclass(frozen=True)
class LineParams:
    """Line detection and surface-call parameters.

    Attributes
    ----------
    band_hz : tuple[float, float]
        Band searched for lines, clipped to Nyquist. Starts above the LFP
        band, whose volume-conducted field grades the profile over hundreds
        of microns.
    n_lines : int
        Strongest peaks kept per block, all entering the surface fit.
    background_hz : tuple[float, float]
        Inner and outer half-widths of the background ring around each bin.
        The inner one also sets how closely two lines may sit and how wide a
        band each line's amplitude integrates, since a wandering line spreads
        over hertz.
    min_excess_db : float
        Across-channel median excess a peak must reach to count as a line.
    widths_um : tuple[float, ...]
        Logistic scales tried on each side of a transition, ``0`` a hard
        step. Capped because a broad enough sigmoid is a ramp, which the
        per-line linear trend absorbs instead.
    min_side : int
        Fewest channels either side of a transition.
    min_separation_um : float
        Closest two transitions may sit.
    max_transitions : int
        Transitions fitted per shank and block. Two because a contact can sit
        in tissue, in the bath, or above it, and line families that couple
        differently step at different depths.
    min_agreeing_lines : int
        Lines that must each support a transition on their own.
    min_delta_bic : float
        Summed support a transition needs beyond its BIC penalty. The
        penalty alone admits in-tissue features, since depth structure
        beyond neighbouring channels is not noise the model knows about.
    outlier_window, outlier_z : int, float
        A channel further than ``outlier_z`` robust deviations from its
        line's running median over ``outlier_window`` channels is replaced by
        that median before fitting; a step survives a running median, an
        isolated bad channel does not.
    dead_ratio, noise_ratio : float
        Channels whose AP RMS falls below or above these multiples of the
        shank median are excluded from the surface fit.
    """

    band_hz: tuple[float, float] = (300.0, 15000.0)
    n_lines: int = 8
    background_hz: tuple[float, float] = (3.0, 25.0)
    min_excess_db: float = 3.0
    widths_um: tuple[float, ...] = (0.0, 7.5, 15.0, 30.0, 60.0, 120.0)
    min_side: int = 7
    min_separation_um: float = 100.0
    max_transitions: int = 2
    min_agreeing_lines: int = 2
    min_delta_bic: float = 500.0
    outlier_window: int = 7
    outlier_z: float = 3.5
    dead_ratio: float = 0.15
    noise_ratio: float = 6.0


@dataclass
class BlockLines:
    """Lines detected in one block and their per-channel amplitudes.

    ``amps_v`` is the rms within ``background_hz[0]`` of each line on each
    channel, background included, so it is positive everywhere and its dB
    profile is defined on channels where the line is buried.
    """

    freqs_hz: np.ndarray  # (n_lines,)
    amps_v: np.ndarray  # (n_lines, n_channels), float32
    excess_db: np.ndarray  # (n_lines,) across-channel median excess
    n_windows: int
    window_s: float


class LineAccumulator:
    """Accumulate each channel's unreferenced Hann power over a block.

    Parameters
    ----------
    n_samples : int
        Window length in samples; windows of any other length are skipped.
    fs : float
        Sampling rate in Hz.
    n_channels : int
        Channels per window.
    params : LineParams
        Band and background settings.
    channel_chunk : int
        Channels handled at once, bounding transient copies.
    """

    def __init__(
        self,
        n_samples: int,
        fs: float,
        n_channels: int,
        params: LineParams,
        channel_chunk: int = 64,
    ) -> None:
        self.n_samples = n_samples
        self.fs = fs
        self.params = params
        self.channel_chunk = channel_chunk
        freqs = scipy.fft.rfftfreq(n_samples, d=1.0 / fs)
        lo, hi = params.band_hz
        in_band = np.flatnonzero((freqs >= lo) & (freqs <= min(hi, fs / 2)))
        # A slice, so the band of a caller's spectrum is a view.
        self.band = slice(int(in_band[0]), int(in_band[-1]) + 1)
        self.freqs = freqs[self.band]
        self.hann = np.hanning(n_samples).astype(np.float32)
        self.power = np.zeros((self.freqs.size, n_channels), dtype=np.float32)
        self.n_windows = 0

    def add_spectrum(self, raw_uv: np.ndarray) -> None:
        """Accumulate one unreferenced ``(n_samples, n_channels)`` window;
        *raw_uv* is not modified."""
        if raw_uv.shape[0] != self.n_samples:
            return
        for start in range(0, raw_uv.shape[1], self.channel_chunk):
            cols = slice(start, start + self.channel_chunk)
            seg = raw_uv[:, cols] - raw_uv[:, cols].mean(axis=0)
            seg *= self.hann[:, None]
            X = np.asarray(scipy.fft.rfft(seg, axis=0))[self.band]
            self.power[:, cols] += X.real**2 + X.imag**2
        self.n_windows += 1

    def add_referenced_spectrum(
        self, X: np.ndarray, reference_uv: np.ndarray
    ) -> None:
        """Accumulate one window from its referenced spectrum.

        *X* is the Hann spectrum of demeaned ``x - reference``; by linearity
        adding the Hann spectrum of the demeaned reference recovers the
        unreferenced one, at the cost of one single-trace transform instead
        of a transform of every channel. Overwrites *X* over the line band.
        """
        if reference_uv.shape[0] != self.n_samples:
            return
        reference = reference_uv - reference_uv.mean()
        M = np.asarray(scipy.fft.rfft(self.hann * reference))[self.band]
        X_band = X[self.band]
        for start in range(0, X_band.shape[1], self.channel_chunk):
            seg = X_band[:, start : start + self.channel_chunk]
            seg += M[:, None]
            self.power[:, start : start + self.channel_chunk] += (
                seg.real**2 + seg.imag**2
            )
        self.n_windows += 1

    def finalize(self) -> BlockLines:
        """Find the strongest lines and measure each on every channel."""
        p = self.params
        n_channels = self.power.shape[1]
        empty = BlockLines(
            freqs_hz=np.zeros(0),
            amps_v=np.zeros((0, n_channels), dtype=np.float32),
            excess_db=np.zeros(0),
            n_windows=self.n_windows,
            window_s=self.n_samples / self.fs,
        )
        df = self.fs / self.n_samples
        inner = max(1, round(p.background_hz[0] / df))
        outer = max(inner + 1, round(p.background_hz[1] / df))
        if self.n_windows == 0 or self.freqs.size <= 2 * outer:
            return empty

        ratios = np.empty_like(self.power)
        for start in range(0, n_channels, self.channel_chunk):
            cols = slice(start, start + self.channel_chunk)
            chunk = self.power[:, cols].astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios[:, cols] = chunk / _ring_mean(chunk, inner, outer)
        excess = np.nanmedian(ratios, axis=1)
        excess[:outer] = excess[-outer:] = np.nan  # ring runs off the band

        peaks = candidate_bins(np.nan_to_num(excess, nan=0.0), inner)
        peaks = peaks[excess[peaks] >= 10 ** (p.min_excess_db / 10)]
        if peaks.size == 0:
            return empty
        peaks = np.sort(peaks[np.argsort(-excess[peaks])][: p.n_lines])

        norm_v2 = 2.0 / (self.n_samples * float(np.sum(self.hann**2)))
        amps = np.empty((peaks.size, n_channels), dtype=np.float32)
        for i, b in enumerate(peaks):
            lobe = self.power[b - inner : b + inner + 1] / self.n_windows
            amps[i] = np.sqrt(norm_v2 * lobe.sum(axis=0, dtype=np.float64))
        return BlockLines(
            freqs_hz=self.freqs[peaks].astype(np.float64),
            amps_v=amps * np.float32(1e-6),
            excess_db=10.0 * np.log10(excess[peaks]),
            n_windows=self.n_windows,
            window_s=self.n_samples / self.fs,
        )


def _ring_mean(power: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Mean over bins between *inner* and *outer* either side, along axis 0."""
    wide, narrow = 2 * outer + 1, 2 * inner + 1
    total = uniform_filter1d(power, wide, axis=0, mode="nearest") * wide
    core = uniform_filter1d(power, narrow, axis=0, mode="nearest") * narrow
    return (total - core) / (wide - narrow)


def candidate_bins(z: np.ndarray, separation: int) -> np.ndarray:
    """Local maxima of *z*, no two within *separation* bins."""
    peak = np.zeros(z.size, dtype=bool)
    peak[1:-1] = (z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:])
    blocked = np.zeros(z.size + 2 * separation, dtype=bool)
    keep: list[int] = []
    for b in np.flatnonzero(peak)[np.argsort(-z[peak], kind="stable")]:
        if not blocked[b + separation]:
            keep.append(int(b))
            blocked[b + 1 : b + 2 * separation] = True
    return np.array(sorted(keep), dtype=np.intp)


# ---------------------------------------------------------------------------
# Surface transitions
# ---------------------------------------------------------------------------
#
# Each line's dB profile along depth is modelled as
#
#     a + c * y + sum_t b_t * logistic((y - y_t) / w_t)
#
# with the transition locations y_t shared by every line and the baseline a,
# trend c, step b_t and widths w_t (one each side of y_t) its own. Lines may
# step by any amount and either sign, or not at all. For fixed locations and
# widths the fit is linear, so the location scan is a set of projections.

_MAD_TO_SD = 1.4826


def usable_channels(rms_ap: np.ndarray, params: LineParams) -> np.ndarray:
    """Channels neither dead nor noisy by AP RMS against their median."""
    ok = np.isfinite(rms_ap) & (rms_ap > 0)
    if not ok.any():
        return ok
    median = float(np.median(rms_ap[ok]))
    return (
        ok
        & (rms_ap >= params.dead_ratio * median)
        & (rms_ap <= params.noise_ratio * median)
    )


def _robust_sd(x: np.ndarray) -> np.ndarray:
    """Scaled median absolute deviation along the last axis."""
    median = np.median(x, axis=-1, keepdims=True)
    return _MAD_TO_SD * np.median(np.abs(x - median), axis=-1)


def _replace_outliers(db: np.ndarray, params: LineParams) -> np.ndarray:
    """Each line's outlying channels replaced by its running median."""
    width = params.outlier_window | 1
    running = median_filter(db, size=(1, width), mode="nearest")
    resid = db - running
    sd = np.maximum(_robust_sd(resid), 1e-6)
    return np.where(
        np.abs(resid) > params.outlier_z * sd[:, None], running, db
    )


def _logistic(
    y: np.ndarray, centre: np.ndarray, below: float, above: float
) -> np.ndarray:
    """``(n_centres, n)`` rise from 0 deep to 1 shallow, scale ``below``
    under the centre and ``above`` over it, a hard step where zero."""
    d = y[None, :] - np.asarray(centre, dtype=float)[:, None]
    scale = np.where(d < 0, below, above)
    with np.errstate(over="ignore"):
        smooth = expit(d / np.where(scale > 0, scale, 1.0))
    return np.where(scale > 0, smooth, (d > 0).astype(float))


def _nuisance_basis(y: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """``(n_lines, n, k)`` orthonormal basis of each line's baseline,
    trend and *fixed* ``(n_lines, n, j)`` transition columns."""
    n_lines, n = fixed.shape[:2]
    trend = (y - y.mean()) / (np.ptp(y) or 1.0)
    base = np.broadcast_to(np.stack([np.ones(n), trend], -1), (n_lines, n, 2))
    return np.linalg.qr(np.concatenate([base, fixed], -1))[0]


def _residualize(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """``x (..., n_lines, n)`` less its projection on each line's basis."""
    coef = np.einsum("...ln,lnk->...lk", x, basis)
    return x - np.einsum("...lk,lnk->...ln", coef, basis)


@dataclass
class _Transition:
    """One shared location and each line's widths either side of it."""

    centre: float
    below_um: np.ndarray  # (n_lines,)
    above_um: np.ndarray

    def columns(self, y: np.ndarray) -> np.ndarray:
        """``(n_lines, n, 1)`` each line's column for this transition."""
        return np.stack(
            [
                _logistic(y, [self.centre], b, a)[0]
                for b, a in zip(self.below_um, self.above_um, strict=True)
            ]
        )[..., None]


def _columns(y: np.ndarray, n_lines: int, fixed: list[_Transition]):
    """Stacked columns of *fixed* transitions, ``(n_lines, n, len)``."""
    if not fixed:
        return np.zeros((n_lines, y.size, 0))
    return np.concatenate([t.columns(y) for t in fixed], -1)


def _scan(
    db: np.ndarray,
    y: np.ndarray,
    fixed: list[_Transition],
    centres: np.ndarray,
    weight: np.ndarray,
    params: LineParams,
) -> tuple[np.ndarray, np.ndarray, _Transition | None, np.ndarray]:
    """Add one transition at each candidate centre on top of *fixed*.

    Returns the weighted gain summed over lines per centre, each line's
    gain at the best centre, that transition, and each line's step there.
    """
    n_lines = db.shape[0]
    basis = _nuisance_basis(y, _columns(y, n_lines, fixed))
    resid = _residualize(db, basis)
    best = np.full((centres.size, n_lines), -np.inf)
    below = np.zeros_like(best)
    above = np.zeros_like(best)
    step = np.zeros_like(best)
    for b in params.widths_um:
        for a in params.widths_um:
            col = _residualize(
                np.broadcast_to(
                    _logistic(y, centres, b, a)[:, None, :],
                    (centres.size, n_lines, y.size),
                ),
                basis,
            )
            norm = np.maximum(np.einsum("mln,mln->ml", col, col), 1e-12)
            proj = np.einsum("mln,ln->ml", col, resid)
            gain = proj**2 / norm
            better = gain > best
            best[better] = gain[better]
            below[better], above[better] = b, a
            step[better] = (proj / norm)[better]
    score = best @ weight
    if not centres.size:
        return score, np.zeros(n_lines), None, np.zeros(n_lines)
    k = int(np.argmax(score))
    return (
        score,
        best[k],
        _Transition(float(centres[k]), below[k], above[k]),
        step[k],
    )


def _line_weights(
    db: np.ndarray, y: np.ndarray, fixed: list[_Transition]
) -> tuple[np.ndarray, float]:
    """Per-line inverse residual variance deflated for correlation between
    neighbouring channels, and the mean effective channel count."""
    basis = _nuisance_basis(y, _columns(y, db.shape[0], fixed))
    resid = _residualize(db, basis)
    sd = np.maximum(_robust_sd(resid), 1e-3)
    centred = resid - resid.mean(-1, keepdims=True)
    lag1 = np.sum(centred[:, 1:] * centred[:, :-1], -1) / np.maximum(
        np.sum(centred**2, -1), 1e-12
    )
    deflate = (1 - np.clip(lag1, 0, 0.99)) / (1 + np.clip(lag1, 0, 0.99))
    return deflate / sd**2, float(y.size * deflate.mean())


def _admissible(
    centres: np.ndarray, others: list[_Transition], params: LineParams
) -> np.ndarray:
    """Centres far enough from every other transition."""
    ok = np.ones(centres.size, dtype=bool)
    for t in others:
        ok &= np.abs(centres - t.centre) >= params.min_separation_um
    return centres[ok]


def _fit(
    db: np.ndarray, y: np.ndarray, centres: np.ndarray, n: int, params
) -> tuple[list[_Transition], np.ndarray]:
    """*n* transitions placed greedily, then each re-placed given the rest."""
    fixed: list[_Transition] = []
    # Channel-to-channel noise: residuals of a fit without the transition
    # would down-weight exactly the lines that have one.
    weight = 2.0 / np.maximum(_robust_sd(np.diff(db, axis=1)), 1e-3) ** 2
    for _ in range(n):
        free = _admissible(centres, fixed, params)
        if not free.size:
            break
        fixed.append(_scan(db, y, fixed, free, weight, params)[2])
        for _ in range(2):
            for i in range(len(fixed)):
                others = fixed[:i] + fixed[i + 1 :]
                free = _admissible(centres, others, params)
                fixed[i] = _scan(db, y, others, free, weight, params)[2]
            weight, _ = _line_weights(db, y, fixed)
    return fixed, weight


def _describe(
    db: np.ndarray,
    y: np.ndarray,
    centres: np.ndarray,
    model: list[_Transition],
    params: LineParams,
) -> list[dict]:
    """Each transition's support given the others, and whether it passes.

    A line supports a transition when adding it, with its own step and two
    widths, beats a BIC penalty. The transition passes when its summed
    support exceeds the penalty for every line plus the shared location by
    ``min_delta_bic`` and ``min_agreeing_lines`` lines support it
    individually.
    """
    weight, n_eff = _line_weights(db, y, model)
    # A step and two widths, the widths also picked from a grid.
    line_penalty = 3 * np.log(max(n_eff, 2.0)) + 2 * np.log(
        len(params.widths_um)
    )
    out = []
    for i, t in enumerate(model):
        others = model[:i] + model[i + 1 :]
        free = _admissible(centres, others, params)
        score, gain, _, _ = _scan(db, y, others, free, weight, params)
        own = _scan(db, y, others, np.array([t.centre]), weight, params)
        support = own[1] * weight
        steps = own[3]
        delta_bic = float(support.sum() - (db.shape[0] * line_penalty))
        delta_bic -= float(np.log(max(n_eff, 2.0)))
        agrees = support > line_penalty
        near = free[score >= score.max() - 3.84] if free.size else free
        spread = np.log(9.0) * (t.below_um + t.above_um)
        out.append(
            {
                "midpoint_um": t.centre,
                "width_um": float(np.median(spread[agrees]))
                if agrees.any()
                else None,
                "interval_um": [float(near.min()), float(near.max())]
                if near.size
                else [t.centre, t.centre],
                "delta_bic": delta_bic,
                "n_agreeing_lines": int(agrees.sum()),
                "n_lines": int(db.shape[0]),
                "passes": bool(
                    delta_bic >= params.min_delta_bic
                    and agrees.sum() >= params.min_agreeing_lines
                ),
                "lines": [
                    {
                        "row": int(row),
                        "step_db": float(steps[row]),
                        "width_below_um": float(t.below_um[row]),
                        "width_above_um": float(t.above_um[row]),
                        "support": float(support[row]),
                        "agrees": bool(agrees[row]),
                    }
                    for row in range(db.shape[0])
                ],
            }
        )
    return out


def fit_transitions(
    amps_v: np.ndarray,
    locations_um: np.ndarray,
    params: LineParams,
    max_transitions: int | None = None,
    min_agreeing_lines: int | None = None,
) -> list[dict]:
    """Brain-surface candidates on one shank from its line profiles.

    Fits up to ``max_transitions`` shared transition locations and keeps the
    largest number whose every transition passes; ``[]`` when none does.
    Which candidate is the brain surface is left to the reader: a second
    boundary, such as the bath surface, can be as strong.

    Parameters
    ----------
    amps_v : np.ndarray
        ``(n_lines, n_channels)`` line rms on the shank's usable channels.
    locations_um : np.ndarray
        ``(n_channels, 2)`` probe-local ``(x, y)``, ``y`` up from the tip.
    params : LineParams
        Fit settings.
    max_transitions, min_agreeing_lines : int or None
        Overrides of the ``params`` values.

    Returns
    -------
    list[dict]
        Per transition, deepest first: ``midpoint_um``; ``width_um``, the
        median 10-90 % extent over agreeing lines; ``interval_um``, the
        midpoints within 3.84 of the best weighted fit; ``delta_bic``;
        ``n_agreeing_lines``; ``n_lines``; and per line ``row``,
        ``step_db`` (shallow minus deep), ``width_below_um``,
        ``width_above_um`` (logistic scales), ``support`` and ``agrees``.
    """
    if min_agreeing_lines is not None or max_transitions is not None:
        params = replace(
            params,
            max_transitions=max_transitions or params.max_transitions,
            min_agreeing_lines=min_agreeing_lines or params.min_agreeing_lines,
        )
    n_lines, n = amps_v.shape
    if n_lines == 0 or n < 2 * params.min_side:
        return []
    order = np.lexsort((locations_um[:, 0], -locations_um[:, 1]))
    y = locations_um[order, 1].astype(float)
    db = 20.0 * np.log10(np.maximum(amps_v[:, order], 1e-30))
    db = _replace_outliers(db.astype(float), params)
    depths = np.unique(y)
    mids = (depths[:-1] + depths[1:]) / 2
    n_above = (y[None, :] > mids[:, None]).sum(1)
    centres = mids[
        (n_above >= params.min_side) & (n - n_above >= params.min_side)
    ]
    if not centres.size:
        return []

    for n_transitions in range(params.max_transitions, 0, -1):
        model, _ = _fit(db, y, centres, n_transitions, params)
        if len(model) < n_transitions:
            continue
        found = _describe(db, y, centres, model, params)
        if all(t["passes"] for t in found):
            return sorted(found, key=lambda t: t["midpoint_um"])
    return []
