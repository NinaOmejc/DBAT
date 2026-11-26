"""Review-only belt quality metrics; no waveform correction or clinical labels."""

from copy import deepcopy

import hashlib
import json

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from ..config import BELTS, window_samples
from .annotations import continuous_runs
from .cleaning import candidate_intervals

ALGORITHM_VERSION = "dbat_signal_quality_v1"
RULES = {
    "reference_windows": 6,
    "mad_multiplier": 6.0,
    "noise_increase_db": 6.0,
    "native_ratio_floor": 0.05,
    "final_ratio_floor": 0.05,
    "native_dominance_ratio": 1.0,
    "final_dominance_ratio": 0.25,
    "typical_energy_fraction": 0.05,
    "instability_floor_db": 10.0,
    "flatline_sec": 1.0,
    "clipping_sec": 0.25,
    "clipping_runs": 3,
    "clipping_fraction": 0.1,
    "constant_fraction": 0.9,
}


def _spectral_energy(values, fs_hz, settings):
    """Estimate band energies on sufficient contiguous finite data using median Welch.

    PSD uses periodic Hann, 50% overlap, constant segment detrending, no zero
    padding and density scaling. Band edges are linearly interpolated before
    trapezoidal integration. Missing data are never filled for this calculation.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    """
    empty = {"respiratory_energy": None, "noise_energy": None, "noise_ratio": None}
    respiratory = settings["respiratory_band_hz"]
    noise = [settings["noise_band_hz"][0], min(settings["noise_band_hz"][1], fs_hz / 2)]
    if respiratory[1] >= fs_hz / 2 or noise[0] >= noise[1]:
        return {**empty, "status": "frequency_band_unavailable"}
    runs = list(continuous_runs(np.isfinite(values)))
    if not runs:
        return {**empty, "status": "missing_data"}
    start, end = max(runs, key=lambda run: run[1] - run[0])
    segment = window_samples(settings["welch_segment_sec"], "seconds", fs_hz)
    if end - start < max(2 * segment, settings["min_finite_fraction"] * len(values)):
        return {**empty, "status": "insufficient_contiguous_data"}
    frequencies, density = welch(
        values[start:end],
        fs=fs_hz,
        window="hann",
        nperseg=segment,
        noverlap=segment // 2,
        detrend="constant",
        scaling="density",
        average="median",
    )
    energies = []
    for lower, upper in (respiratory, noise):
        axis = np.r_[
            lower, frequencies[(frequencies > lower) & (frequencies < upper)], upper
        ]
        energies.append(float(trapezoid(np.interp(axis, frequencies, density), axis)))
    if not np.isfinite(energies).all():
        return {**empty, "status": "nonfinite_energy"}
    respiratory_energy, noise_energy = energies
    ratio = noise_energy / respiratory_energy if respiratory_energy > 0 else None
    if ratio is not None and not np.isfinite(ratio):
        ratio = None
    return {
        "respiratory_energy": respiratory_energy,
        "noise_energy": noise_energy,
        "noise_ratio": ratio,
        "status": "assessed"
        if respiratory_energy > 0
        else "undefined_respiratory_energy",
    }


def _plateau_masks(values, fs_hz):
    """Find exact digital plateaus, not low-amplitude breathing or isolated peaks.

    Flatline means identical finite values for >=1 s. Inferred rail clipping
    requires >=3 separate >=0.25 s plateaus at the same recording extreme.
    An isolated flat-topped excursion is never sufficient clipping evidence.
    """
    finite = np.isfinite(values)
    flat, clipped = np.zeros(len(values), bool), np.zeros(len(values), bool)
    if not finite.any():
        return flat, clipped, False
    low, high = np.min(values[finite]), np.max(values[finite])
    rails = {float(low): [], float(high): []}
    equal = finite[1:] & finite[:-1] & (values[1:] == values[:-1])
    for first, last in continuous_runs(equal):
        start, end = int(first), int(last + 1)
        if end - start >= RULES["flatline_sec"] * fs_hz:
            flat[start:end] = True
        if (
            low != high
            and end - start >= RULES["clipping_sec"] * fs_hz
            and values[start] in rails
        ):
            rails[float(values[start])].append((start, end))
    for runs in rails.values():
        if len(runs) >= RULES["clipping_runs"]:
            for start, end in runs:
                clipped[start:end] = True
    return flat, clipped, bool(finite.all() and low == high)


def _limit(values, floor, *, minimum_increment=0.0):
    """Return a conservative median + six scaled MAD threshold, or undefined."""
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if len(finite) < RULES["reference_windows"]:
        return None
    centre = np.median(finite)
    return float(
        max(
            floor,
            centre
            + max(
                minimum_increment,
                RULES["mad_multiplier"] * 1.4826 * np.median(abs(finite - centre)),
            ),
        )
    )


def assess_signal_quality(
    recording, cleaned, final, processed_fs_hz, settings, signature,
    *, previous=None, changed=None
):
    """Measure native noise and residual final usability without changing samples.

    Parameters
    ----------
    recording : Recording
        Defines native identity, channel order and the annotation grid.
    cleaned, final : ndarray
        Polarity-oriented cleaned native and final processed waveforms. Read only.
    processed_fs_hz : float
        Final sampling rate in Hz; final HF band is capped at its Nyquist.
    settings : dict
        Window/step, bands, Welch segment, finite coverage and review flag settings.
    signature : str
        Source plus numerical-preprocessing identity, including waveform decisions.
    previous : dict, optional
        Cached quality metrics; unchanged channels reuse their windows and summary.
    changed : set of str, optional
        Channels with changed numerical inputs, checked by the caller.

    Returns
    -------
    dict
        Per-window metrics, channel summaries, thresholds, provenance and native
        review candidates. Undefined values are None. No combined score, signal
        modification, baseline/spike detector or automatic exclusion is applied.
    """
    result = {
        "algorithm_version": ALGORITHM_VERSION,
        "settings": dict(settings),
        "native_fs_hz": recording.fs_hz,
        "processed_fs_hz": processed_fs_hz,
        "windows": [],
        "channels": {},
        "candidates": [],
        "fixed_parameters": dict(RULES),
        "policy": "Numerical finite support, independent of manual approval; interpolated values describe the actual cleaned waveform; review-only flags",
    }
    result["definitions"] = {
        "hf_noise_ratio": "Native cleaned HF band energy / respiratory band energy",
        "final_hf_noise_ratio": "Same ratio on final waveform; HF upper edge capped at processed Nyquist",
        "noise_instability_db": "Absolute 10*log10(current native HF energy / previous window HF energy); first/undefined neighbour is null",
        "flatline_fraction": "Fraction of finite native samples in exact >=1 s plateaus; never flagged by itself",
        "clipping_fraction": "Fraction in >=3 repeated >=0.25 s plateaus at the same recording extreme; inferred, not known ADC rails",
        "edges": "Complete native windows only; short recordings have undefined window metrics; no gap filling; longest finite run must cover min_finite_fraction and >=2 Welch segments",
        "spectral": "Median Welch; periodic Hann, 50% overlap, constant detrending, no FFT padding; trapezoidal band integration with interpolated edges",
        "thresholds": "Experimental review heuristics: >=6 reference windows, 6 scaled MAD; amplitude alone is never a noise trigger",
    }
    if not settings["enabled"]:
        result["status"] = "disabled"
        return result
    result["status"] = "assessed"
    window = window_samples(settings["window_sec"], "seconds", recording.fs_hz)
    step = window_samples(settings["step_sec"], "seconds", recording.fs_hz)
    count = len(recording.samples)
    starts = list(range(0, max(0, count - window + 1), step)) or [0]
    result["unassessed_tail_sec"] = (
        count - min(count, starts[-1] + window)
    ) / recording.fs_hz
    result["frequency_resolution_hz"] = {
        "native": recording.fs_hz
        / window_samples(settings["welch_segment_sec"], "seconds", recording.fs_hz),
        "processed": processed_fs_hz
        / window_samples(settings["welch_segment_sec"], "seconds", processed_fs_hz),
    }
    processed_times = np.arange(len(final)) / processed_fs_hz
    candidate_mask = np.zeros(recording.samples.shape, bool)
    events = []
    for column, channel in enumerate(recording.channels):
        if channel not in BELTS:
            continue
        if previous is not None and channel not in changed:
            result["channels"][channel] = deepcopy(previous["channels"][channel])
            result["windows"].extend(deepcopy([
                row for row in previous["windows"] if row["channel"] == channel
            ]))
            continue
        flat, clipped, constant = _plateau_masks(cleaned[:, column], recording.fs_hz)
        rows, previous_noise = [], None
        for start in starts:
            end = min(count, start + window)
            lower, upper = start / recording.fs_hz, end / recording.fs_hz
            first, last = np.searchsorted(processed_times, [lower, upper])
            native = cleaned[start:end, column]
            processed = final[first:last, column]
            full = end - start == window
            raw = (
                _spectral_energy(native, recording.fs_hz, settings)
                if full
                else {
                    "status": "insufficient_duration",
                    "noise_energy": None,
                    "respiratory_energy": None,
                    "noise_ratio": None,
                }
            )
            post = (
                _spectral_energy(processed, processed_fs_hz, settings)
                if full
                else dict(raw)
            )
            finite = np.isfinite(native)
            sufficient = full and finite.mean() >= settings["min_finite_fraction"]
            noise = raw["noise_energy"]
            instability = (
                abs(10 * np.log10(noise / previous_noise))
                if noise is not None
                and previous_noise is not None
                and noise > 0
                and previous_noise > 0
                else None
            )
            row = {
                "channel": channel,
                "start_sample": start,
                "end_sample": end,
                "start_sec": lower,
                "end_sec": upper,
                "processed_start_sample": int(first),
                "processed_end_sample": int(last),
                "input_status": raw["status"],
                "final_status": post["status"],
                "finite_fraction": float(finite.mean()),
                "final_finite_fraction": float(np.isfinite(processed).mean())
                if len(processed)
                else 0.0,
                "hf_noise_ratio": raw["noise_ratio"],
                "noise_energy": noise,
                "respiratory_energy": raw["respiratory_energy"],
                "final_hf_noise_ratio": post["noise_ratio"],
                "final_noise_energy": post["noise_energy"],
                "final_respiratory_energy": post["respiratory_energy"],
                "noise_instability_db": float(instability)
                if instability is not None
                else None,
                "flatline_fraction": float(flat[start:end][finite].mean())
                if sufficient
                else None,
                "clipping_fraction": float(clipped[start:end][finite].mean())
                if sufficient
                else None,
                "flatline_or_clipping_fraction": float(
                    (flat[start:end] | clipped[start:end])[finite].mean()
                )
                if sufficient
                else None,
                "flagged": False,
                "reasons": [],
            }
            rows.append(row)
            previous_noise = noise
        powers = [
            10 * np.log10(row["noise_energy"])
            if row["noise_energy"] is not None and row["noise_energy"] > 0
            else None
            for row in rows
        ]
        ratios = [
            10 * np.log10(row["hf_noise_ratio"])
            if row["hf_noise_ratio"] is not None and row["hf_noise_ratio"] > 0
            else None
            for row in rows
        ]
        power_limit = _limit(
            powers, -np.inf, minimum_increment=RULES["noise_increase_db"]
        )
        ratio_limit = _limit(ratios, 10 * np.log10(RULES["native_ratio_floor"]))
        instability_limit = _limit(
            [row["noise_instability_db"] for row in rows], RULES["instability_floor_db"]
        )
        resp = [
            row["respiratory_energy"]
            for row in rows
            if row["respiratory_energy"] is not None and row["respiratory_energy"] > 0
        ]
        typical_resp = float(np.median(resp)) if resp else None
        for index, row in enumerate(rows):
            ratio, final_ratio, noise = (
                row["hf_noise_ratio"],
                row["final_hf_noise_ratio"],
                row["noise_energy"],
            )
            remaining = (
                final_ratio is not None and final_ratio > RULES["final_ratio_floor"]
            )
            enough_energy = (
                noise is not None
                and typical_resp is not None
                and noise > RULES["typical_energy_fraction"] * typical_resp
            )
            relative = (
                remaining
                and enough_energy
                and power_limit is not None
                and ratio_limit is not None
                and powers[index] is not None
                and powers[index] > power_limit
                and ratios[index] is not None
                and ratios[index] > ratio_limit
            )
            dominant = (
                remaining
                and enough_energy
                and ratio is not None
                and ratio > RULES["native_dominance_ratio"]
                and final_ratio > RULES["final_dominance_ratio"]
            )
            if relative or dominant:
                row["reasons"].append("residual_high_frequency_noise")
            if (
                remaining
                and enough_energy
                and power_limit is not None
                and powers[index] is not None
                and powers[index] > power_limit
                and instability_limit is not None
                and row["noise_instability_db"] is not None
                and row["noise_instability_db"] > instability_limit
            ):
                row["reasons"].append("unstable_noise_energy")
            if (
                row["clipping_fraction"] is not None
                and row["clipping_fraction"] >= RULES["clipping_fraction"]
            ):
                row["reasons"].append("possible_sensor_clipping")
            if (
                constant
                and row["flatline_fraction"] is not None
                and row["flatline_fraction"] >= RULES["constant_fraction"]
            ):
                row["reasons"].append("constant_waveform")
            row["flagged"] = bool(row["reasons"])
        defined = [row for row in rows if row["flatline_fraction"] is not None]

        def median_metric(key):
            numbers = [row[key] for row in rows if row[key] is not None]
            return float(np.median(numbers)) if numbers else None

        finite_powers = [value for value in powers if value is not None]
        result["channels"][channel] = {
            "window_count": len(rows),
            "assessed_windows": len(defined),
            "median_noise_ratio": median_metric("hf_noise_ratio"),
            "median_final_noise_ratio": median_metric("final_hf_noise_ratio"),
            "median_noise_instability_db": median_metric("noise_instability_db"),
            "noise_variability_db": float(
                1.4826
                * np.median(abs(np.asarray(finite_powers) - np.median(finite_powers)))
            )
            if finite_powers
            else None,
            "flagged_window_fraction": sum(row["flagged"] for row in defined)
            / len(defined)
            if defined
            else None,
            "thresholds": {
                "native_noise_energy_db": power_limit,
                "native_noise_ratio_db": ratio_limit,
                "instability_db": instability_limit,
                "typical_respiratory_energy": typical_resp,
            },
        }
        result["windows"].extend(rows)
    for row in result["windows"]:
        channel = row["channel"]
        column = recording.channels.index(channel)
        if row["flagged"] and settings["flag_candidates"]:
            candidate_mask[row["start_sample"] : row["end_sample"], column] = True
            for reason in row["reasons"]:
                events.append(
                    {
                        "channel": channel,
                        "action": "review_candidate",
                        "reason": reason,
                        "start_sample": row["start_sample"],
                        "end_sample": row["end_sample"],
                        "start_sec": row["start_sec"],
                        "end_sec": row["end_sec"],
                        "metrics": {
                            key: row[key]
                            for key in (
                                "hf_noise_ratio",
                                "final_hf_noise_ratio",
                                "noise_instability_db",
                                "flatline_fraction",
                                "clipping_fraction",
                            )
                        },
                    }
                )
    stage_signature = hashlib.sha256(
        json.dumps([ALGORITHM_VERSION, signature, settings], sort_keys=True).encode()
    ).hexdigest()
    result["signature"] = stage_signature
    result["candidates"] = candidate_intervals(
        recording,
        {"auto_candidate": candidate_mask},
        events,
        stage_signature,
        source_stage="signal_quality",
    )
    return result
