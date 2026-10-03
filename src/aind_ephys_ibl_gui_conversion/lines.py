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
    min_edge_um : float
        Least recorded depth either side of a transition. The contacts at
        the ends of a shank couple to the lines differently from the rest,
        so a step within this of an end is an edge, not a boundary.
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
    dead_line_db : float
        Channels whose median line level sits this far below the shank's
        are excluded too: AP RMS misses a channel that carries noise but
        none of the lines, and adjacent ones survive outlier replacement.
    """

    band_hz: tuple[float, float] = (300.0, 15000.0)
    n_lines: int = 8
    background_hz: tuple[float, float] = (3.0, 25.0)
    min_excess_db: float = 3.0
    widths_um: tuple[float, ...] = (0.0, 7.5, 15.0, 30.0, 60.0, 120.0)
    min_side: int = 7
    min_edge_um: float = 100.0
    min_separation_um: float = 100.0
    max_transitions: int = 2
    min_agreeing_lines: int = 2
    min_delta_bic: float = 500.0
    outlier_window: int = 7
    outlier_z: float = 3.5
    dead_ratio: float = 0.15
    noise_ratio: float = 6.0
    dead_line_db: float = 40.0


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

    def _ring(self) -> tuple[int, int]:
        """Inner and outer background half-widths in bins."""
        df = self.fs / self.n_samples
        inner = max(1, round(self.params.background_hz[0] / df))
        outer = max(inner + 1, round(self.params.background_hz[1] / df))
        return inner, outer

    def excess_spectrum(self) -> np.ndarray | None:
        """Across-channel median of each bin's power over its background
        ring, ``NaN`` where the ring runs off the band; ``None`` without
        data."""
        inner, outer = self._ring()
        if self.n_windows == 0 or self.freqs.size <= 2 * outer:
            return None
        n_channels = self.power.shape[1]
        ratios = np.empty_like(self.power)
        for start in range(0, n_channels, self.channel_chunk):
            cols = slice(start, start + self.channel_chunk)
            chunk = self.power[:, cols].astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios[:, cols] = chunk / _ring_mean(chunk, inner, outer)
        excess = np.nanmedian(ratios, axis=1)
        excess[:outer] = excess[-outer:] = np.nan
        return excess

    def measure(
        self, bins: np.ndarray, excess: np.ndarray | None
    ) -> BlockLines:
        """Each line at *bins* measured on every channel; *excess* is this
        block's excess spectrum."""
        inner, _ = self._ring()
        n_channels = self.power.shape[1]
        norm_v2 = 2.0 / (self.n_samples * float(np.sum(self.hann**2)))
        amps = np.empty((bins.size, n_channels), dtype=np.float32)
        for i, b in enumerate(bins):
            lobe = self.power[b - inner : b + inner + 1] / self.n_windows
            amps[i] = np.sqrt(norm_v2 * lobe.sum(axis=0, dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            excess_db = (
                10.0 * np.log10(excess[bins])
                if excess is not None
                else np.full(bins.size, np.nan)
            )
        return BlockLines(
            freqs_hz=self.freqs[bins].astype(np.float64),
            amps_v=amps * np.float32(1e-6),
            excess_db=excess_db,
            n_windows=self.n_windows,
            window_s=self.n_samples / self.fs,
        )

    def finalize(self) -> BlockLines:
        """Find this block's strongest lines and measure each on every
        channel."""
        found = pool_lines([self], self.params)[0]
        assert found is not None
        return found


def pool_lines(
    accumulators: list[LineAccumulator | None], params: LineParams
) -> list[BlockLines | None]:
    """Lines found across blocks and measured in every block.

    A line counts when it stands out in any block, by the largest of the
    blocks' excess spectra, so each block's amplitude profile covers the
    same lines whether or not that block sees them. Blocks on different
    frequency grids are pooled separately.
    """
    out: list[BlockLines | None] = [None] * len(accumulators)
    groups: dict[tuple[int, float], list[int]] = {}
    for i, acc in enumerate(accumulators):
        if acc is not None:
            groups.setdefault((acc.n_samples, acc.fs), []).append(i)
    for members in groups.values():
        accs = [accumulators[i] for i in members]
        spectra = [acc.excess_spectrum() for acc in accs]  # type: ignore[union-attr]
        live = [e for e in spectra if e is not None]
        bins = np.zeros(0, dtype=np.intp)
        if live:
            with np.errstate(invalid="ignore"):
                pooled = np.fmax.reduce(live)
            inner, _ = accs[0]._ring()  # type: ignore[union-attr]
            peaks = candidate_bins(np.nan_to_num(pooled, nan=0.0), inner)
            peaks = peaks[pooled[peaks] >= 10 ** (params.min_excess_db / 10)]
            bins = np.sort(peaks[np.argsort(-pooled[peaks])][: params.n_lines])
        for i, acc, excess in zip(members, accs, spectra, strict=True):
            out[i] = acc.measure(  # type: ignore[union-attr]
                bins if excess is not None else bins[:0], excess
            )
    return out


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


def _segments(blocks: np.ndarray) -> list[slice]:
    """Runs of equal *blocks*, which is sorted."""
    edges = np.flatnonzero(np.diff(blocks)) + 1
    bounds = np.r_[0, edges, blocks.size]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def _replace_outliers(
    db: np.ndarray, segments: list[slice], params: LineParams
) -> np.ndarray:
    """Each line's outlying channels replaced by its running median, both
    taken within each block so neither an offset nor a noisier block
    counts against another."""
    width = params.outlier_window | 1
    out = db.copy()
    for seg in segments:
        running = median_filter(db[:, seg], size=(1, width), mode="nearest")
        resid = db[:, seg] - running
        sd = np.maximum(_robust_sd(resid), 1e-6)
        outlying = np.abs(resid) > params.outlier_z * sd[:, None]
        out[:, seg] = np.where(outlying, running, db[:, seg])
    return out


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


@dataclass
class _Profile:
    """Depth-sorted channels of one shank pooled over blocks."""

    y: np.ndarray  # (n,) um from the tip
    blocks: np.ndarray  # (n,) block of each channel, sorted
    segments: list[slice]
    base: np.ndarray  # (n, n_blocks + 1) block baselines and shared trend

    @classmethod
    def build(cls, y: np.ndarray, blocks: np.ndarray) -> _Profile:
        """Profile of channels already sorted by block."""
        levels = np.unique(blocks)
        indicators = (blocks[:, None] == levels[None, :]).astype(float)
        trend = (y - y.mean()) / (np.ptp(y) or 1.0)
        return cls(
            y, blocks, _segments(blocks), np.c_[indicators, trend[:, None]]
        )


def _nuisance_basis(
    profile: _Profile, fixed: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """``(n_lines, n, k)`` orthonormal basis of each line's whitened block
    baselines, trend and *fixed* ``(n_lines, n, j)`` transition columns;
    *scale* ``(n_lines, n)`` whitens each channel."""
    base = np.broadcast_to(profile.base, (fixed.shape[0], *profile.base.shape))
    columns = np.concatenate([base, fixed], -1)
    return np.linalg.qr(columns * scale[..., None])[0]


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
    profile: _Profile,
    fixed: list[_Transition],
    centres: np.ndarray,
    noise: _Noise,
    params: LineParams,
) -> tuple[np.ndarray, np.ndarray, _Transition | None, np.ndarray]:
    """Add one transition at each candidate centre on top of *fixed*.

    Returns the whitened gain summed over lines per centre; each line's
    gain at the best centre; that transition; and each
    line's step there in dB.
    """
    n_lines, n = db.shape
    y = profile.y
    scale = noise.scale
    basis = _nuisance_basis(profile, _columns(y, n_lines, fixed), scale)
    resid = _residualize(db * scale, basis)
    # A candidate column c enters only through inner products with each
    # line's whitened residual, whitened basis and whitening, so every
    # centre is scanned by three matrix products instead of residualizing
    # an (n_centres, n_lines, n) array. The column depends on depth alone,
    # so channels sharing a depth are summed first.
    k = basis.shape[-1]
    depths, at_depth = np.unique(y, return_inverse=True)
    to_depth = np.zeros((depths.size, n))
    to_depth[at_depth, np.arange(n)] = 1.0
    against_resid = to_depth @ (scale * resid).T  # (n_depths, n_lines)
    against_basis = to_depth @ (scale[..., None] * basis).transpose(
        1, 0, 2
    ).reshape(n, n_lines * k)
    against_self = to_depth @ (scale**2).T
    best = np.full((centres.size, n_lines), -np.inf)
    below = np.zeros_like(best)
    above = np.zeros_like(best)
    step = np.zeros_like(best)
    for b in params.widths_um:
        for a in params.widths_um:
            c = _logistic(depths, centres, b, a)  # (n_centres, n_depths)
            proj = c @ against_resid
            on_basis = (c @ against_basis).reshape(-1, n_lines, k)
            norm = (c**2) @ against_self - np.sum(on_basis**2, -1)
            norm = np.maximum(norm, 1e-12)
            gain = proj**2 / norm
            better = gain > best
            best[better] = gain[better]
            below[better], above[better] = b, a
            step[better] = (proj / norm)[better]
    score = best.sum(-1)
    if not centres.size:
        return score, np.zeros(n_lines), None, np.zeros(n_lines)
    k = int(np.argmax(score))
    return (
        score,
        best[k],
        _Transition(float(centres[k]), below[k], above[k]),
        step[k],
    )


@dataclass
class _Noise:
    """Per-channel whitening, deflated per line and block for correlation
    between neighbouring channels."""

    scale: np.ndarray  # (n_lines, n)
    n_eff: float

    @classmethod
    def from_neighbours(cls, db: np.ndarray, profile: _Profile) -> _Noise:
        """Channel-to-channel noise per line and block; residuals of a fit
        without a transition would inflate it for exactly the lines that
        have one."""
        scale = np.empty_like(db)
        for seg in profile.segments:
            sd = _robust_sd(np.diff(db[:, seg], axis=1)) / np.sqrt(2.0)
            scale[:, seg] = (1.0 / np.maximum(sd, 1e-3))[:, None]
        return cls(scale, float(profile.y.size))

    @classmethod
    def from_fit(
        cls,
        db: np.ndarray,
        profile: _Profile,
        fixed: list[_Transition],
        previous: _Noise,
    ) -> _Noise:
        """Residual sd per line and block given *fixed*, each block's
        weight deflated by its own lag-1 correlation so one correlated block
        does not discount the others."""
        basis = _nuisance_basis(
            profile, _columns(profile.y, db.shape[0], fixed), previous.scale
        )
        resid = _residualize(db * previous.scale, basis) / previous.scale
        scale = np.empty_like(db)
        n_eff = 0.0
        for seg in profile.segments:
            white = resid[:, seg]
            sd = np.maximum(_robust_sd(white), 1e-3)
            white = white / sd[:, None]
            white -= white.mean(-1, keepdims=True)
            lag1 = np.sum(white[:, 1:] * white[:, :-1], -1) / np.maximum(
                np.sum(white**2, -1), 1e-12
            )
            rho = np.clip(lag1, 0, 0.99)
            deflate = (1 - rho) / (1 + rho)
            scale[:, seg] = (np.sqrt(deflate) / sd)[:, None]
            n_eff += white.shape[1] * float(deflate.mean())
        return cls(scale, n_eff)


def _admissible(
    centres: np.ndarray, others: list[_Transition], params: LineParams
) -> np.ndarray:
    """Centres far enough from every other transition."""
    ok = np.ones(centres.size, dtype=bool)
    for t in others:
        ok &= np.abs(centres - t.centre) >= params.min_separation_um
    return centres[ok]


def _fit(
    db: np.ndarray,
    profile: _Profile,
    centres: np.ndarray,
    n: int,
    params: LineParams,
) -> tuple[list[_Transition], _Noise]:
    """*n* transitions placed greedily, then each re-placed given the rest."""
    fixed: list[_Transition] = []
    noise = _Noise.from_neighbours(db, profile)
    for _ in range(n):
        free = _admissible(centres, fixed, params)
        if not free.size:
            break
        fixed.append(_scan(db, profile, fixed, free, noise, params)[2])
        for _ in range(2):
            for i in range(len(fixed)):
                others = fixed[:i] + fixed[i + 1 :]
                free = _admissible(centres, others, params)
                fixed[i] = _scan(db, profile, others, free, noise, params)[2]
            noise = _Noise.from_fit(db, profile, fixed, noise)
    return fixed, noise


def _coefficients(
    db: np.ndarray, profile: _Profile, model: list[_Transition], noise: _Noise
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each line's block baselines at the tip (dB), trend (dB per mm) and
    steps (dB) with every transition in *model* fitted jointly."""
    n_blocks = profile.base.shape[1] - 1
    cols = _columns(profile.y, db.shape[0], model)
    baselines = np.empty((db.shape[0], n_blocks))
    trend = np.empty(db.shape[0])
    steps = np.empty((db.shape[0], len(model)))
    for line in range(db.shape[0]):
        design = np.c_[
            profile.base[:, :n_blocks], profile.y / 1000.0, cols[line]
        ]
        w = noise.scale[line]
        coef = np.linalg.lstsq(design * w[:, None], db[line] * w, rcond=None)[
            0
        ]
        baselines[line] = coef[:n_blocks]
        trend[line] = coef[n_blocks]
        steps[line] = coef[n_blocks + 1 :]
    return baselines, trend, steps


def _describe(
    db: np.ndarray,
    profile: _Profile,
    centres: np.ndarray,
    model: list[_Transition],
    noise: _Noise,
    params: LineParams,
) -> list[dict]:
    """Each transition's support given the others, and whether it passes.

    A line supports a transition when adding it, with its own step and two
    widths, beats a BIC penalty. ``delta_bic`` sums what the supporting
    lines gain beyond that penalty, less the shared location's, so lines
    that do not step cost nothing and the gate does not tighten with the
    number of lines. The transition passes on ``min_delta_bic`` with
    ``min_agreeing_lines`` supporting lines.
    """
    n_eff = noise.n_eff
    # A step and two widths, the widths also picked from a grid.
    line_penalty = 3 * np.log(max(n_eff, 2.0)) + 2 * np.log(
        len(params.widths_um)
    )
    baselines, trend, steps = _coefficients(db, profile, model, noise)
    blocks = np.unique(profile.blocks)
    out = []
    for i, t in enumerate(model):
        others = model[:i] + model[i + 1 :]
        free = _admissible(centres, others, params)
        score, _, _, _ = _scan(db, profile, others, free, noise, params)
        own = _scan(db, profile, others, np.array([t.centre]), noise, params)
        support = own[1]
        agrees = support > line_penalty
        delta_bic = float(np.sum(support[agrees] - line_penalty))
        delta_bic -= float(np.log(max(n_eff, 2.0)))
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
                "pinned": _pinned(t.centre, free, profile),
                "lines": [
                    {
                        "row": int(row),
                        "step_db": float(steps[row, i]),
                        "width_below_um": float(t.below_um[row]),
                        "width_above_um": float(t.above_um[row]),
                        "support": float(support[row]),
                        "agrees": bool(agrees[row]),
                        "baseline_db": {
                            int(b): float(v)
                            for b, v in zip(
                                blocks, baselines[row], strict=True
                            )
                        },
                        "trend_db_per_mm": float(trend[row]),
                    }
                    for row in range(db.shape[0])
                ],
            }
        )
    return out


def _pinned(centre: float, free: np.ndarray, profile: _Profile) -> bool:
    """Whether *centre* is the last admissible one inside some block: the
    data then place it no further from that block's end, so it is the end's
    offset rather than a located boundary."""
    for seg in profile.segments:
        y = profile.y[seg]
        inside = free[(free > y.min()) & (free < y.max())]
        if inside.size and centre in (inside.min(), inside.max()):
            return True
    return False


def _identifiable_centres(
    y: np.ndarray, profile: _Profile, params: LineParams
) -> np.ndarray:
    """Midpoints between depths with ``min_side`` channels and
    ``min_edge_um`` of one block on each side. A transition no block
    straddles is a block offset, so a boundary between blocks that tile the
    shank is never a candidate."""
    depths = np.unique(y)
    mids = (depths[:-1] + depths[1:]) / 2
    ok = np.zeros(mids.size, dtype=bool)
    for seg in profile.segments:
        above = (y[seg][None, :] > mids[:, None]).sum(1)
        ok |= (
            (above >= params.min_side)
            & (y[seg].size - above >= params.min_side)
            & (mids - y[seg].min() >= params.min_edge_um)
            & (y[seg].max() - mids >= params.min_edge_um)
        )
    return mids[ok]


def _carries_lines(db: np.ndarray, params: LineParams) -> np.ndarray:
    """Channels whose median level over lines, each line referenced to its
    shank median, is within ``dead_line_db`` of the shank's."""
    level = np.median(db - np.median(db, axis=1, keepdims=True), axis=0)
    return level > -params.dead_line_db


def fit_transitions(
    amps_v: np.ndarray,
    locations_um: np.ndarray,
    params: LineParams,
    blocks: np.ndarray | None = None,
    max_transitions: int | None = None,
    min_agreeing_lines: int | None = None,
) -> list[dict]:
    """Brain-surface candidates on one shank from its line profiles.

    Fits up to ``max_transitions`` shared transition locations and keeps the
    largest number whose every transition passes; ``[]`` when none does.
    Which candidate is the brain surface is left to the reader: a second
    boundary, such as the bath surface, can be as strong.

    Channels may come from several blocks. A line's level drifts between
    recordings by several dB, so each line has its own baseline per block;
    its trend, steps and widths are shared.

    Parameters
    ----------
    amps_v : np.ndarray
        ``(n_lines, n_channels)`` line rms on the shank's usable channels.
    locations_um : np.ndarray
        ``(n_channels, 2)`` probe-local ``(x, y)``, ``y`` up from the tip.
    params : LineParams
        Fit settings.
    blocks : np.ndarray or None
        ``(n_channels,)`` block of each channel; one block when ``None``.
    max_transitions, min_agreeing_lines : int or None
        Overrides of the ``params`` values.

    Returns
    -------
    list[dict]
        Per transition, deepest first: ``midpoint_um``; ``width_um``, the
        median 10-90 % extent over agreeing lines; ``interval_um``, the
        midpoints within 3.84 of the best whitened fit; ``delta_bic``;
        ``n_agreeing_lines``; ``n_lines``; and per line ``row``,
        ``step_db`` (shallow minus deep), ``width_below_um``,
        ``width_above_um`` (logistic scales), ``support``, ``agrees``, and
        the joint fit's ``baseline_db`` per block at the tip and
        ``trend_db_per_mm``.
    """
    if min_agreeing_lines is not None or max_transitions is not None:
        params = replace(
            params,
            max_transitions=max_transitions or params.max_transitions,
            min_agreeing_lines=min_agreeing_lines or params.min_agreeing_lines,
        )
    if blocks is None:
        blocks = np.zeros(amps_v.shape[1], dtype=int)
    if amps_v.shape[0]:
        live = _carries_lines(
            20.0 * np.log10(np.maximum(amps_v, 1e-30)), params
        )
        amps_v, locations_um = amps_v[:, live], locations_um[live]
        blocks = np.asarray(blocks)[live]
    n_lines, n = amps_v.shape
    if n_lines == 0 or n < 2 * params.min_side:
        return []
    order = np.lexsort((locations_um[:, 0], -locations_um[:, 1], blocks))
    y = locations_um[order, 1].astype(float)
    profile = _Profile.build(y, np.asarray(blocks)[order])
    db = 20.0 * np.log10(np.maximum(amps_v[:, order], 1e-30))
    db = _replace_outliers(db.astype(float), profile.segments, params)
    centres = _identifiable_centres(y, profile, params)
    if not centres.size:
        return []

    for n_transitions in range(params.max_transitions, 0, -1):
        model, noise = _fit(db, profile, centres, n_transitions, params)
        if len(model) < n_transitions:
            continue
        found = _describe(db, profile, centres, model, noise, params)
        # A pinned transition stays in the model to absorb the edge but is
        # neither required to pass nor reported.
        located = [t for t in found if not t.pop("pinned")]
        if all(t["passes"] for t in located):
            return sorted(located, key=lambda t: t["midpoint_um"])
    return []
