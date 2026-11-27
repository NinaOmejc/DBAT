"""Joint-belt balance, lognormal FFT phase and averaged-spectra coherence.

Numerical adaptation of Nina Omejc's MIT-licensed MAGMA-ml implementation:
https://github.com/NinaOmejc/magma-ml/blob/815b01a7a52baac087d504308fa6d633e98ab984/src/features/thoracoabdominal.py
MATLAB comparison:
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/label_detection/compute_respiratory_phase_offset_metrics.m
Copyright (c) 2026 Nina Omejc. See LICENSE. Lognormal FFT phase is a named
Python alternative to MATLAB wtI, not an asserted exact port of that transform.
"""

import numpy as np
from scipy import signal, stats

from ..preprocessing.annotations import continuous_runs
from .common import (
    band_power,
    observations,
    reference_value,
    spectral_windows,
    trailing_support,
)


def balance_measurements(valid, fs, belts, references, cfg, selected):
    """Thirty-second trailing independently session-normalized amplitude medians."""
    names = (
        "thoracic_amplitude_median",
        "abdominal_amplitude_median",
        "ta_ratio",
        "ta_log_ratio",
        "thoracic_fraction",
    )
    times, rows, coverage, counts = [], [], [], []
    for t, _, _, complete, fraction in trailing_support(
        valid, fs, cfg["step_sec"], cfg["balance_window_sec"]
    ):
        values, numbers = [], []
        for channel in ("Resp-Lungs", "Resp-Diaphragm"):
            belt = belts[channel]
            ratio = belt["amp"] / reference_value(references[channel]["session"])
            keep = (
                (belt["peak_t"] >= t - cfg["balance_window_sec"])
                & (belt["peak_t"] <= t)
                & belt["breath_valid"]
                & np.isfinite(ratio)
                & (ratio > 0)
            )
            numbers.append(int(keep.sum()))
            values.append(
                np.median(ratio[keep])
                if complete and keep.sum() >= cfg["balance_min_breaths"]
                else np.nan
            )
        a, b = values
        ratio = a / b if np.isfinite([a, b]).all() and min(a, b) > 0 else np.nan
        rows.append((a, b, ratio, np.log(ratio) if ratio > 0 else np.nan, a / (a + b)))
        times.append(t)
        coverage.append(fraction)
        counts.append(numbers)
    times, rows, counts = (
        np.asarray(times),
        np.asarray(rows).reshape(-1, 5),
        np.asarray(counts).reshape(-1, 2),
    )
    return {
        name: observations(
            rows[:, i],
            times,
            start=times - cfg["balance_window_sec"],
            end=times,
            coverage=coverage,
            resolution="trailing_window",
            n_valid_breaths=counts,
            minimum_breaths_per_belt=cfg["balance_min_breaths"],
        )
        for i, name in enumerate(names)
        if name in selected
    }


def _lognormal_wavelet(values, fs, frequency, f0, band):
    """Discrete lognormal analytic wavelet with 99% impulse-energy edge support.

    Same lognormal frequency response as wtI, cubic detrending and hard passband.
    Zero padding and discrete energy-support estimation differ from wtI's
    Padding=none and adaptive integral support. This is a NAMED ALTERNATIVE.
    No claim of numerical equivalence is made.
    """
    n = len(values)
    time = np.arange(n) / fs
    design = np.polynomial.polynomial.polyvander(time / max(time[-1], 1 / fs), 3)
    x = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    f_native = np.fft.fftfreq(n, 1 / fs)
    fx = np.fft.fft(x)
    fx[(abs(f_native) <= max(band[0], fs / n)) | (abs(f_native) >= band[1])] = 0
    x = np.fft.ifft(fx).real
    # >=8 cycles of the lowest frequency on each side bounds wraparound.
    padding = int(np.ceil(8 * f0 * fs / frequency[0]))
    nfft = int(2 ** np.ceil(np.log2(n + 2 * padding)))
    fft_frequency = np.fft.fftfreq(nfft, 1 / fs)
    positive = fft_frequency > 0
    fx = np.fft.fft(x, nfft)
    result = np.full((len(frequency), n), np.nan + 1j * np.nan)
    distances = np.minimum(np.arange(nfft), nfft - np.arange(nfft))
    q = 2 * np.pi * f0
    for i, center in enumerate(frequency):
        response = np.zeros(nfft)
        response[positive] = np.exp(
            -0.5 * q**2 * np.log(fft_frequency[positive] / center) ** 2
        )
        impulse = abs(np.fft.ifft(response)) ** 2
        energy = np.bincount(distances, weights=impulse)
        edge = int(np.searchsorted(np.cumsum(energy), 0.99 * energy.sum()))
        if 2 * edge >= n:
            continue
        coefficients = np.fft.ifft(fx * response)[:n]
        result[i, edge : n - edge] = coefficients[edge : n - edge]
    return result


def phase_measurements(values, valid, fs, belts, cfg, step, selected):
    """Shared-frequency, reviewed-IBI-guided circular means within valid runs."""
    lower, upper = cfg["band_hz"]
    voices = int(
        np.ceil(10 * np.log(2) * 2 * np.pi * cfg["f0"] / (2 * stats.norm.ppf(0.75)))
    )
    frequency = 2 ** (
        np.arange(
            np.ceil(voices * np.log2(lower)), np.floor(voices * np.log2(upper)) + 1
        )
        / voices
    )
    grid = np.arange(0, (len(valid) - 1) / fs + 1e-9, step)
    names = ("phase_difference", "absolute_phase_difference", "phase_consistency")
    rows = np.full((len(grid), 3), np.nan)
    starts, ends = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    reliable = np.zeros(len(grid), bool)
    for first, stop in continuous_runs(valid):
        n = stop - first
        if n / fs < (cfg["summary_cycles"] + 2 * cfg["edge_cycles"]) / upper:
            continue
        normalized = []
        for x in values:
            a = x[first:stop] - np.median(x[first:stop])
            scale = 1.4826 * np.median(abs(a))
            if scale <= np.finfo(float).eps:
                scale = np.std(a, ddof=1)
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                break
            normalized.append(a / scale)
        if len(normalized) != 2:
            continue
        local_t = np.arange(n) / fs
        absolute_t = first / fs + local_t
        timing = []
        for channel in ("Resp-Lungs", "Resp-Diaphragm"):
            belt = belts[channel]
            expected = np.full(n, np.nan)
            for i, ok in enumerate(belt["ibi_valid"]):
                a, b = belt["peak_idx"][i : i + 2]
                f = 1 / belt["ibi"][i] if ok and belt["ibi"][i] > 0 else np.nan
                if lower <= f <= upper and first <= a < b < stop:
                    expected[(absolute_t >= a / fs) & (absolute_t <= b / fs)] = f
            timing.append(expected)
        expected = np.where(
            np.isfinite(timing[0]) & np.isfinite(timing[1]),
            (timing[0] + timing[1]) / 2,
            np.where(np.isfinite(timing[0]), timing[0], timing[1]),
        )
        w1, w2 = (
            _lognormal_wavelet(x, fs, frequency, cfg["f0"], (lower, upper))
            for x in normalized
        )
        joint = np.sqrt(abs(w1) * abs(w2))
        finite = np.isfinite(joint)
        tol = cfg["frequency_tolerance_fraction"]
        guided = (
            finite
            & (frequency[:, None] >= (1 - tol) * expected)
            & (frequency[:, None] <= (1 + tol) * expected)
        )
        eligible = np.where(guided.any(axis=0)[None, :], guided, finite)
        row = np.argmax(np.where(eligible, joint, -np.inf), axis=0)
        columns = np.arange(n)
        a, b = w1[row, columns], w2[row, columns]
        picked = frequency[row]
        floor_frequency = (
            max(lower, (1 - tol) * np.nanmin(expected))
            if np.isfinite(expected).any()
            else lower
        )
        edge = cfg["edge_cycles"] / floor_frequency
        support = (
            eligible.any(axis=0) & (local_t >= edge) & (local_t <= local_t[-1] - edge)
        )
        floors = [
            max(
                np.finfo(float).eps,
                cfg["min_magnitude_fraction"]
                * np.median(np.max(np.where(np.isfinite(w), abs(w), 0), axis=0)),
            )
            for w in (w1, w2)
        ]
        usable = support & (abs(a) >= floors[0]) & (abs(b) >= floors[1])
        phase = np.angle(a * np.conj(b))
        for g in np.flatnonzero((grid >= first / fs) & (grid < stop / fs)):
            center = int(np.floor((grid[g] - first / fs) * fs + 0.5))
            if center >= n or not usable[center]:
                continue
            half = int(np.ceil(0.5 * cfg["summary_cycles"] * fs / picked[center]))
            lo, hi = center - half, center + half + 1
            if (
                lo < 0
                or hi > n
                or not support[lo:hi].all()
                or usable[lo:hi].mean() < cfg["min_valid_fraction"]
            ):
                continue
            z = np.mean(np.exp(1j * phase[lo:hi][usable[lo:hi]]))
            angle = np.degrees(np.angle(z))
            rows[g] = (angle, abs(angle), abs(z))
            starts[g], ends[g] = (first + lo) / fs, (first + hi) / fs
            reliable[g] = abs(z) >= cfg["min_resultant_length"]
    return {
        name: observations(
            rows[:, i],
            grid,
            start=starts,
            end=ends,
            resolution="centered_cycle_window",
            reliable=reliable,
            method="lognormal_fft_phase",
            matlab_wtI_equivalent=False,
        )
        for i, name in enumerate(names)
        if name in selected
    }


def coherence_measurement(values, valid, fs, cfg):
    """Average valid 60-s/30-s Hann cross-spectra before forming coherence."""
    count = int(np.floor(cfg["segment_sec"] * fs + 0.5))
    freq = np.fft.rfftfreq(count, 1 / fs)
    segments = []
    opts = {
        "fs": fs,
        "window": "hann",
        "nperseg": count,
        "noverlap": 0,
        "nfft": count,
        "detrend": "constant",
        "scaling": "density",
    }
    for first, stop in spectral_windows(
        valid, fs, cfg["segment_sec"], cfg["segment_step_sec"]
    ):
        a, b = (x[first:stop] for x in values)
        segments.append(
            (
                first / fs,
                stop / fs,
                signal.welch(a, **opts)[1],
                signal.welch(b, **opts)[1],
                signal.csd(a, b, **opts)[1],
            )
        )
    rows, starts, counts = [], [], []
    for first in np.arange(
        0, max(0, len(valid) / fs - cfg["window_sec"]) + 1e-9, cfg["step_sec"]
    ):
        if first + cfg["window_sec"] > len(valid) / fs:
            continue
        keep = [
            s for s in segments if s[0] >= first and s[1] <= first + cfg["window_sec"]
        ]
        value = np.nan
        if len(keep) >= cfg["min_segments"]:
            pxx, pyy, pxy = (np.mean([s[i] for s in keep], axis=0) for i in (2, 3, 4))
            denominator = pxx * pyy
            c = np.divide(
                abs(pxy) ** 2,
                denominator,
                out=np.full(len(freq), np.nan),
                where=denominator > 0,
            )
            c = np.clip(c, 0, 1)
            band = (freq >= cfg["band_hz"][0]) & (freq <= cfg["band_hz"][1])
            if np.isfinite(c[band]).all():
                value = np.clip(
                    band_power(freq, c, cfg["band_hz"])
                    / (cfg["band_hz"][1] - cfg["band_hz"][0]),
                    0,
                    1,
                )
        rows.append(value)
        starts.append(first)
        counts.append(len(keep))
    starts = np.asarray(starts)
    return observations(
        rows,
        starts + cfg["window_sec"] / 2,
        start=starts,
        end=starts + cfg["window_sec"],
        resolution="spectral_window",
        n_valid_segments=np.asarray(counts),
        minimum_segments=cfg["min_segments"],
        bias_warning="Segment count affects coherence bias",
    )
