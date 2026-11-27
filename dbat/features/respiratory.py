"""Reviewed breath, local, envelope, spectral and eAMI measurements.

Adapted from Nina Omejc's MIT-licensed MAGMA-ml src/features/respiratory.py:
https://github.com/NinaOmejc/magma-ml/blob/815b01a7a52baac087d504308fa6d633e98ab984/src/features/respiratory.py
eAMI and P95-P05 equations from the MATLAB sources:
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/label_detection/compute_eami_periodic_breathing.m
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/utils/robust_resp_excursion.m
Copyright (c) 2026 Nina Omejc. See LICENSE. No detection or reference calculation.
"""

from fractions import Fraction

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


def breath_measurements(belt, reference):
    """Reuse saved amplitudes/IBIs; derive only trough-to-peak/peak-to-trough times."""
    peaks, troughs = belt["peak_t"], belt["trough_t"]
    amp = np.where(belt["breath_valid"] & (belt["amp"] > 0), belt["amp"], np.nan)
    ibi = np.where(belt["ibi_valid"], belt["ibi"], np.nan)
    n = len(peaks)
    inspiratory, expiratory = np.full(n, np.nan), np.full(n, np.nan)
    if n > 1:
        inspiratory[1:] = np.where(belt["trough_valid"], peaks[1:] - troughs, np.nan)
        expiratory[:-1] = np.where(belt["trough_valid"], troughs - peaks[:-1], np.nan)
    inspiratory[~belt["breath_valid"]] = np.nan
    expiratory[~belt["breath_valid"]] = np.nan
    out = {
        "ibi": observations(
            ibi, peaks[:-1], start=peaks[:-1], end=peaks[1:], resolution="inter_breath"
        ),
        "respiratory_rate": observations(
            np.where(belt["ibi_valid"], belt["rr_bpm"], np.nan),
            peaks[:-1],
            start=peaks[:-1],
            end=peaks[1:],
            resolution="inter_breath",
        ),
    }
    for name, values in (
        ("amplitude", amp),
        ("amp_ratio", amp / reference_value(reference["session"])),
        ("global_amp_ratio", amp / reference_value(reference["global"])),
        ("inspiratory_duration", inspiratory),
        ("expiratory_duration", expiratory),
        ("ie_ratio", inspiratory / np.maximum(expiratory, 0.1)),
    ):
        # Saved amplitude support is the adjacent trough pair; undefined edges stay missing.
        start = np.r_[np.nan, troughs] if n else np.array([])
        end = np.r_[troughs, np.nan] if n else np.array([])
        method = belt["settings"]["amp_method"]
        if name == "inspiratory_duration" or (
            name in {"amplitude", "amp_ratio", "global_amp_ratio"}
            and method == "inspiratory"
        ):
            end = peaks
        elif name == "expiratory_duration" or (
            name in {"amplitude", "amp_ratio", "global_amp_ratio"}
            and method == "expiratory"
        ):
            start = peaks
        out[name] = observations(
            values, peaks, start=start, end=end, resolution="breath"
        )
    return out


def local_measurements(values, valid, fs, belt, reference, cfg, selected):
    """Trailing complete-IBI rates/CoV and QC-covered local waveform excursion."""
    out, step = {}, cfg["step_sec"]
    p, ibis = belt["peak_t"], np.where(belt["ibi_valid"], belt["ibi"], np.nan)
    for window, names, minimum in (
        (cfg["rr_window_sec"], ("local_respiratory_rate",), cfg["rr_min_ibi"]),
        (cfg["cov_window_sec"], ("ibi_cov", "ibi_robust_cov"), cfg["cov_min_ibi"]),
        (cfg["excursion_window_sec"], ("waveform_excursion", "excursion_ratio"), 3),
    ):
        if not set(names) & selected:
            continue
        times, counts, coverage = [], [], []
        rows = {name: [] for name in names}
        for t, first, stop, complete, fraction in trailing_support(
            valid, fs, step, window
        ):
            times.append(t)
            coverage.append(fraction)
            if "waveform_excursion" in names:
                x = values[first:stop]
                x = x[np.isfinite(x)]
                value = (
                    np.diff(np.percentile(x, [5, 95], method="hazen"))[0]
                    if (
                        t >= window
                        and len(x) >= 3
                        and fraction >= cfg["excursion_min_coverage"]
                    )
                    else np.nan
                )
                counts.append(len(x))
                rows["waveform_excursion"].append(value)
                rows["excursion_ratio"].append(
                    value / reference_value(reference["waveform"], "excursion")
                )
            else:
                a = ibis[
                    (p[:-1] >= t - window)
                    & (p[1:] <= t)
                    & np.isfinite(ibis)
                    & (ibis > 0)
                ]
                counts.append(len(a) if complete else 0)
                ok = complete and len(a) >= minimum
                if names == ("local_respiratory_rate",):
                    rows[names[0]].append(60 / np.mean(a) if ok else np.nan)
                else:
                    rows["ibi_cov"].append(
                        np.std(a, ddof=1) / np.mean(a) if ok else np.nan
                    )
                    med = np.median(a) if ok else np.nan
                    rows["ibi_robust_cov"].append(
                        1.4826 * np.median(abs(a - med)) / med if ok else np.nan
                    )
        times = np.asarray(times)
        for name in set(names) & selected:
            out[name] = observations(
                rows[name],
                times,
                start=times - window,
                end=times,
                coverage=coverage,
                n_valid=np.asarray(counts),
                minimum_count=minimum,
                resolution="trailing_window",
            )
    return out


def envelope_measurements(normalized, valid, fs, cfg, selected):
    """Hilbert envelope per valid run; local population-SD/mean CoV, matching MAGMA-ml."""
    envelope = np.full(len(normalized), np.nan)
    for first, stop in continuous_runs(valid & np.isfinite(normalized)):
        if stop - first >= 2:
            envelope[first:stop] = abs(signal.hilbert(normalized[first:stop]))
    time = np.arange(len(envelope)) / fs
    out = {}
    if "envelope" in selected:
        out["envelope"] = observations(
            envelope,
            time,
            end=time + 1 / fs,
            resolution="sample",
            transform_context="full contiguous QC run; median/IQR normalization",
        )
    if "envelope_cov" in selected:
        rows = list(
            trailing_support(
                valid & np.isfinite(envelope),
                fs,
                cfg["step_sec"],
                cfg["envelope_window_sec"],
            )
        )
        times = np.array([r[0] for r in rows])
        values = [
            np.std(envelope[a:b]) / np.mean(envelope[a:b])
            if ok and np.mean(envelope[a:b]) > 0
            else np.nan
            for _, a, b, ok, _ in rows
        ]
        out["envelope_cov"] = observations(
            values,
            times,
            start=times - cfg["envelope_window_sec"],
            end=times,
            coverage=[r[4] for r in rows],
            resolution="trailing_window",
        )
    return out


def spectral_measurements(values, valid, fs, belt, cfg, selected):
    """Native Hann/Welch windows; fixed-width power bands and reviewed-IBI harmonic."""
    names = (
        "dominant_frequency",
        "spectral_bandwidth",
        "spectral_entropy",
        "dominant_power_fraction",
        "second_harmonic_power_fraction",
    )
    starts, ends, rows, counts = [], [], [], []
    for first, stop in spectral_windows(valid, fs, cfg["window_sec"], cfg["step_sec"]):
        f, psd = signal.welch(
            values[first:stop],
            fs,
            window="hann",
            nperseg=stop - first,
            nfft=stop - first,
        )
        in_band = (f >= cfg["band_hz"][0]) & (f <= cfg["band_hz"][1])
        freq, power = f[in_band], psd[in_band]
        row = np.full(5, np.nan)
        complete = (
            (belt["peak_t"][:-1] >= first / fs)
            & (belt["peak_t"][1:] < stop / fs)
            & belt["ibi_valid"]
        )
        ibis = belt["ibi"][complete]
        ibis = ibis[np.isfinite(ibis) & (ibis > 0)]
        if len(freq) >= 2 and power.sum() > 0:
            probability = power / power.sum()
            peak = freq[np.argmax(power)]
            row[:3] = (
                peak,
                np.sqrt(np.sum(probability * (freq - np.sum(freq * probability)) ** 2)),
                stats.entropy(probability) / np.log(len(power)),
            )
            total = band_power(f, psd, cfg["band_hz"])
            if np.isfinite(total) and total > 0:
                harmonic = (
                    2 / np.median(ibis)
                    if len(ibis) >= cfg["harmonic_min_ibi"]
                    else np.nan
                )
                for i, center in ((3, peak), (4, harmonic)):
                    if (
                        np.isfinite(center)
                        and cfg["band_hz"][0] <= center <= cfg["band_hz"][1]
                    ):
                        band = (
                            max(cfg["band_hz"][0], center - cfg["peak_half_width_hz"]),
                            min(cfg["band_hz"][1], center + cfg["peak_half_width_hz"]),
                        )
                        row[i] = np.clip(band_power(f, psd, band) / total, 0, 1)
        rows.append(row)
        starts.append(first / fs)
        ends.append(stop / fs)
        counts.append(len(ibis))
    start, end = np.asarray(starts), np.asarray(ends)
    rows = np.asarray(rows).reshape(-1, 5)
    return {
        name: observations(
            rows[:, i],
            (start + end) / 2,
            start=start,
            end=end,
            resolution="spectral_window",
            n_valid_ibi=np.asarray(counts),
            peak_half_width_hz=cfg["peak_half_width_hz"],
        )
        for i, name in enumerate(names)
        if name in selected
    }


def eami_measurement(values, valid, fs, cfg):
    """MATLAB eAMI: 1 - 0.5*log(var(resp)/var(lowpassed rectified resp)).

    Feature-specific filters/resampling operate independently on valid runs.
    SciPy SOS filtfilt/polyphase edges need not equal MATLAB's implementations.
    """
    band = signal.butter(
        cfg["bandpass_order"] // 2,
        cfg["band_hz"],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    rate = cfg["sampling_hz"]
    low = signal.butter(cfg["lowpass_order"], cfg["lowpass_hz"], fs=rate, output="sos")
    ratio = Fraction(rate / fs).limit_denominator(100000)
    count = int(np.floor(cfg["energy_window_sec"] * rate + 0.5)) + 1
    times, starts, ends, result = [], [], [], []
    pad = 3 * (2 * len(band) + 1)
    lowpad = 3 * (2 * len(low) + 1)
    for first, stop in continuous_runs(valid):
        if stop - first <= pad:
            continue
        filtered = signal.sosfiltfilt(band, values[first:stop])
        filtered = signal.resample_poly(filtered, ratio.numerator, ratio.denominator)
        # Remove polyphase outputs beyond the final observed processed sample.
        local_t = first / fs + np.arange(len(filtered)) / rate
        filtered = filtered[local_t <= (stop - 1) / fs + 1e-10]
        local_t = local_t[: len(filtered)]
        if len(filtered) <= lowpad or len(filtered) < count:
            continue
        envelope = signal.sosfiltfilt(low, abs(filtered))
        for i in range(len(filtered) - count + 1):
            resp, am = filtered[i : i + count], envelope[i : i + count]
            er, ea = (
                np.mean((resp - np.mean(resp)) ** 2),
                np.mean((am - np.mean(am)) ** 2),
            )
            result.append(1 - 0.5 * np.log(er / ea) if er > 0 and ea > 0 else np.nan)
            times.append(local_t[i + (count - 1) // 2])
            starts.append(local_t[i])
            ends.append(local_t[i + count - 1] + 1 / rate)
    return observations(
        result,
        times,
        start=starts,
        end=ends,
        resolution="centered_energy_window",
        sampling_hz=rate,
        method="MATLAB eAMI equation; SciPy finite-run filters/polyphase",
    )
