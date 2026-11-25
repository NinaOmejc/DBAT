"""Conservative automatic corrections; candidates never establish manual QC.

Adapted from the user-supplied test_cleaning.py and test_spo2_cleaning.py.
Their SHA-256 hashes and effective numerical parameters are saved in provenance.
Only interpolation is applied; experimental gap and baseline-correction views
are deliberately absent. SciPy supplies median/Gaussian filters and peak finding.
"""

import hashlib
import json

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from ..config import BELTS
from ..logging import logger
from .annotations import continuous_runs, recording_identity, utc_now
from .polarity import assess_relative_belt_polarity

BELT_SOURCE_SHA256 = "4e066170f49fa51fbbe0629b3edfb79a8c4f3c1b24dbcbeb9ce9c6acc0ad7328"
SPO2_SOURCE_SHA256 = "e5596768f8835634fafb4388edf53af89ae91040530e5693b28c9bafc3bf5968"
ALGORITHM_VERSION = "dbat_auto_cleaning_v2"


def robust_scale(values):
    """Return median absolute deviation scaled to standard-deviation units."""
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def odd_samples(seconds, fs_hz):
    """Return an odd window, at least three samples, for a duration in seconds."""
    samples = max(3, round(seconds * fs_hz))
    return samples + (samples % 2 == 0)


def _baseline(values, fs_hz, seconds):
    """Estimate a temporary robust baseline using smoothed one-second medians."""
    block = max(1, round(fs_hz))
    reduced = np.array([np.median(values[i : i + block]) for i in range(0, len(values), block)])
    trend = median_filter(reduced, size=odd_samples(seconds, 1.0), mode="nearest")
    trend = gaussian_filter1d(trend, sigma=2.0, mode="nearest")
    return np.interp(np.arange(len(values)), np.arange(len(reduced)) * block, trend)


def _estimation_signal(values):
    """Fill nonfinite samples temporarily for detection; never return as cleaned data."""
    finite = np.isfinite(values)
    index = np.arange(len(values))
    return np.interp(index, index[finite], values[finite])


def paired_breath_candidates(belts, fs_hz):
    """Locate paired peaks for an optional interpolation safeguard, not physiology.

    Parameters
    ----------
    belts : dict of ndarray
        Native belt vectors. Missing samples are filled only for detection.
    fs_hz : float
        Native sampling rate in Hz.

    Returns
    -------
    ndarray of int
        Paired event centers on the native grid; these are not sigh labels.
    """
    peaks = {}
    for name in BELTS:
        values = belts[name]
        if np.isfinite(values).sum() < 2 or len(values) < max(3, round(fs_hz)):
            return np.array([], dtype=int)
        smooth = gaussian_filter1d(_estimation_signal(values), sigma=0.10 * fs_hz)
        detrended = smooth - _baseline(smooth, fs_hz, 20)
        threshold = max(
            5 * robust_scale(detrended), 2 * np.percentile(np.abs(detrended), 95), 1e-10
        )
        locations, _ = find_peaks(
            detrended,
            prominence=threshold,
            distance=max(1, round(fs_hz)),
            width=(0.6 * fs_hz, 8 * fs_hz),
        )
        peaks[name] = locations
    pairs = []
    for first in peaks[BELTS[0]]:
        matches = peaks[BELTS[1]][np.abs(peaks[BELTS[1]] - first) <= 1.5 * fs_hz]
        if matches.size:
            second = matches[np.argmin(np.abs(matches - first))]
            center = int((first + second) // 2)
            if not pairs or center - pairs[-1] > 2 * fs_hz:
                pairs.append(center)
    return np.array(pairs, dtype=int)


def _event(start, end, fs_hz, action, reason):
    """Describe an automatic half-open native interval without manual evidence."""
    return {
        "start_sample": int(start),
        "end_sample": int(end),
        "start_sec": start / fs_hz,
        "end_sec": end / fs_hz,
        "action": action,
        "reason": reason,
    }


def preprocess_belt_artifacts(values, fs_hz, paired_times, maximum_sec):
    """Interpolate only brief residual spikes with finite, compatible flanks.

    Returns corrected values, interpolation/candidate masks, events and detector
    parameters. Optional paired peaks remain protected. Baseline-displacement
    detection is absent; detrending owns baseline correction. Detection is adapted
    from the latest supplied test_cleaning.py;
    the fixed 50-sample flank lag is replaced with 0.25 * fs_hz.
    """
    corrected = values.copy()
    interpolated = np.zeros(len(values), dtype=bool)
    candidates = np.zeros(len(values), dtype=bool)
    events = []
    info = {"status": "insufficient_finite_samples"}
    finite = np.isfinite(values)
    lag = max(1, round(0.25 * fs_hz))
    if finite.sum() < 2 or len(values) <= lag:
        return corrected, interpolated, candidates, events, info
    safe = _estimation_signal(values)
    residual = safe - median_filter(safe, size=odd_samples(0.40, fs_hz), mode="nearest")
    threshold = max(10 * robust_scale(residual), 4 * np.percentile(np.abs(residual), 99), 1e-10)
    seeds = finite & (np.abs(residual) > threshold)
    initial = binary_dilation(seeds, iterations=max(1, round(0.035 * fs_hz))) & finite
    groups = []
    for start, end in continuous_runs(initial):
        if groups and start - groups[-1][1] <= 0.6 * fs_hz:
            groups[-1] = (groups[-1][0], end)
        else:
            groups.append((start, end))
    typical_change = np.percentile(np.abs(safe[lag:] - safe[:-lag]), 95)
    flank_limit = max(3 * np.percentile(np.abs(residual), 95), 2 * typical_change, 1e-10)
    for start, end in groups:
        near_pair = np.any(
            (paired_times > start - round(fs_hz)) & (paired_times < end + round(fs_hz))
        )
        if near_pair:
            reason = "paired_peak_protection"
        elif (end - start) / fs_hz > maximum_sec:
            reason = "too_long_to_reconstruct"
        elif start == 0 or end >= len(values) or not finite[start - 1 : end + 1].all():
            reason = "missing_or_boundary_flank"
        elif abs(values[start - 1] - values[end]) > flank_limit:
            reason = "incompatible_flanks"
        else:
            corrected[start:end] = np.interp(
                np.arange(start, end), [start - 1, end], [values[start - 1], values[end]]
            )
            interpolated[start:end] = True
            events.append(_event(start, end, fs_hz, "interpolated", "brief_isolated_spike"))
            continue
        candidates[start:end] = True
        events.append(_event(start, end, fs_hz, "review_candidate", reason))
    candidates &= finite & ~interpolated
    info = {
        "status": "assessed",
        "residual_threshold": float(threshold),
        "flank_limit": float(flank_limit),
    }
    return corrected, interpolated, candidates, events, info


def preprocess_spo2(values, fs_hz, settings):
    """Interpolate abrupt, reversible SpO2 artifacts in declared percentage units.

    Parameters
    ----------
    values : ndarray, shape (n_samples,)
        Native SpO2 values; never modified, smoothed, detrended or clipped.
    fs_hz : float
        Native sampling rate in Hz.
    settings : dict
        Automatic-cleaning settings with percentage-point thresholds and seconds.

    Returns
    -------
    tuple
        One corrected vector, interpolation/candidate masks, events and provenance.
        Adapted from supplied test_spo2_cleaning.py; long flat artifacts are now
        interpolated under the configured duration limit, rather than made NaN.
    """
    corrected = values.copy()
    interpolated = np.zeros(len(values), dtype=bool)
    candidates = np.zeros(len(values), dtype=bool)
    events = []
    finite = np.isfinite(values)
    info = {"status": "insufficient_finite_samples", "units_verified": False}
    if finite.sum() < 2 or len(values) < max(3, round(0.3 * fs_hz)):
        return corrected, interpolated, candidates, events, info
    median = float(np.median(values[finite]))
    low, high = np.percentile(values[finite], [5, 95])
    if settings["spo2_units"] != "percent" or not (50 <= median <= 100 and 0 <= low <= high <= 100):
        info["status"] = "skipped_units_not_verified_as_percent"
        return corrected, interpolated, candidates, events, info
    info = {
        "status": "assessed",
        "units_verified": True,
        "units_basis": "declared_percent_and_range_check",
    }
    safe = _estimation_signal(values)
    smooth = median_filter(safe, size=max(3, 2 * round(0.03 * fs_hz) + 1))
    lag = max(2, round(0.15 * fs_hz))
    changes = smooth[lag:] - smooth[:-lag]
    threshold = settings["spo2_change_threshold_pp"]
    tolerance = settings["spo2_return_tolerance_pp"]
    common = {
        "height": threshold,
        "prominence": threshold / 2,
        "distance": max(1, round(0.7 * fs_hz)),
    }
    rising, _ = find_peaks(changes, **common)
    falling, _ = find_peaks(-changes, **common)
    edges = sorted(
        [(int(i + lag // 2), 1) for i in rising] + [(int(i + lag // 2), -1) for i in falling]
    )
    near, far = max(1, round(0.3 * fs_hz)), max(2, round(1.5 * fs_hz))
    padding = max(1, round(0.12 * fs_hz))
    consumed_edges = set()
    for edge_index, ((start, direction), (end, reverse)) in enumerate(zip(edges[:-1], edges[1:])):
        # A recovery edge cannot also start a second, inverted artifact event.
        if edge_index in consumed_edges or edge_index + 1 in consumed_edges:
            continue
        if (
            direction != -reverse
            or not 2 <= end - start <= settings["spo2_max_interpolation_sec"] * fs_hz
        ):
            continue
        lo, hi = max(0, start - padding), min(len(values), end + padding)
        left_start, left_end = max(0, start - far), max(0, start - near)
        right_start, right_end = min(len(values), end + near), min(len(values), end + far)
        left, right = values[left_start:left_end], values[right_start:right_end]
        reason = None
        if (
            lo == 0
            or hi >= len(values)
            or (hi - lo) / fs_hz > settings["spo2_max_interpolation_sec"]
        ):
            reason = "boundary_or_duration_limit"
        elif min(left.size, right.size) < max(5, round(0.3 * fs_hz)):
            reason = "insufficient_flanks"
        elif (
            not finite[lo - 1 : hi + 1].all()
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
        ):
            reason = "nonfinite_event_or_flanks"
        else:
            before, after = float(np.median(left)), float(np.median(right))
            inside = float(np.median(safe[start:end]))
            excursion = direction * (
                inside - (min(before, after) if direction == 1 else max(before, after))
            )
            if abs(before - after) > tolerance or abs(values[lo - 1] - values[hi]) > tolerance:
                reason = "does_not_return_to_baseline"
            elif excursion < threshold:
                reason = "insufficient_excursion"
            elif any(
                np.percentile(flank, 95) - np.percentile(flank, 5) > tolerance
                for flank in (left, right)
            ):
                reason = "unstable_flanks"
            elif end - start > round(0.25 * fs_hz):
                interior = safe[start + near : end - near]
                if (
                    interior.size < max(5, round(0.25 * fs_hz))
                    or np.percentile(interior, 95) - np.percentile(interior, 5) > 1.0
                ):
                    reason = "nonflat_long_event"
        if reason:
            candidates[lo:hi] = True
            events.append(_event(lo, hi, fs_hz, "review_candidate", reason))
        else:
            corrected[lo:hi] = np.interp(
                np.arange(lo, hi), [lo - 1, hi], [values[lo - 1], values[hi]]
            )
            interpolated[lo:hi] = True
            consumed_edges.update((edge_index, edge_index + 1))
            events.append(_event(lo, hi, fs_hz, "interpolated", "abrupt_reversible_spo2_artifact"))
    candidates &= ~interpolated
    return corrected, interpolated, candidates, events, info


def candidate_intervals(recording, masks, events, signature, *, source_stage):
    """Consolidate per-channel warning masks while retaining all detector evidence.

    Each disjoint native candidate retains stable identity, reasons and original
    events. Interpolated samples are excluded by the existing detector masks;
    consolidation does not change correction thresholds or breath protection.
    """
    candidates = []
    for column, name in enumerate(recording.channels):
        for start, end in continuous_runs(masks["auto_candidate"][:, column]):
            evidence = [
                event
                for event in events
                if event["channel"] == name
                and event["action"] == "review_candidate"
                and event["start_sample"] < end
                and event["end_sample"] > start
            ]
            reasons = list(dict.fromkeys(event["reason"] for event in evidence))
            key = json.dumps([signature, source_stage, name, int(start), int(end)])
            candidates.append(
                {
                    "id": hashlib.sha256(key.encode()).hexdigest()[:24],
                    "channel": name,
                    "source_stage": source_stage,
                    "preparation_signature": signature,
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "start_sec": start / recording.fs_hz,
                    "end_sec": end / recording.fs_hz,
                    "reasons": reasons,
                    "evidence": evidence,
                }
            )
    return sorted(candidates, key=lambda item: (item["start_sample"], item["channel"]))


def clean_recording(recording, config):
    """Detect/correct each supported channel once, retaining immutable native arrays.

    Returns a preparation dictionary containing one corrected matrix, automatic
    masks/events/provenance and a source/settings signature for compatible review.
    Manual ratings, inspected coverage and confirmed artifacts are never inferred.
    """
    settings = config["preprocessing"]["artifacts"]
    corrected = recording.samples.copy()
    masks = {
        name: np.zeros(corrected.shape, dtype=bool) for name in ("interpolated", "auto_candidate")
    }
    events, channel_info = [], {}
    belts = {name: recording.samples[:, recording.channels.index(name)] for name in BELTS}
    paired = (
        paired_breath_candidates(belts, recording.fs_hz)
        if settings["automatic_detection"] and settings["paired_breath_protection"]
        else np.array([], dtype=int)
    )
    for column, name in enumerate(recording.channels):
        if not settings["automatic_detection"] or name not in (*BELTS, "SpO2"):
            channel_info[name] = {
                "status": "disabled" if not settings["automatic_detection"] else "not_targeted"
            }
            continue
        values = recording.samples[:, column]
        if name in BELTS:
            output = preprocess_belt_artifacts(
                values, recording.fs_hz, paired, settings["belt_max_interpolation_sec"]
            )
        else:
            output = preprocess_spo2(values, recording.fs_hz, settings)
        (
            corrected[:, column],
            masks["interpolated"][:, column],
            masks["auto_candidate"][:, column],
            detected,
            info,
        ) = output
        channel_info[name] = info
        events.extend({"channel": name, **event} for event in detected)
    parameters = {
        "belt_residual_window_sec": 0.40,
        "belt_mad_multiplier": 10,
        "belt_abs_p99_multiplier": 4,
        "belt_dilation_sec": 0.035,
        "belt_group_gap_sec": 0.6,
        "belt_flank_lag_sec": 0.25,
        "paired_gaussian_sigma_sec": 0.10,
        "paired_baseline_sec": 20,
        "paired_width_sec": [0.6, 8],
        "paired_tolerance_sec": 1.5,
        "paired_protection_margin_sec": 1.0,
        "baseline_block_sec": 1.0,
        "baseline_gaussian_sigma_blocks": 2.0,
        "paired_mad_multiplier": 5,
        "paired_abs_p95_multiplier": 2,
        "paired_min_separation_sec": 2,
        "belt_flank_abs_p95_multiplier": 3,
        "belt_flank_change_multiplier": 2,
        "spo2_detection_median_half_window_sec": 0.03,
        "spo2_same_direction_edge_separation_sec": 0.7,
        "spo2_edge_prominence_fraction": 0.5,
        "spo2_brief_event_sec": 0.25,
        "spo2_edge_window_sec": 0.15,
        "spo2_edge_padding_sec": 0.12,
        "spo2_long_plateau_spread_pp": 1.0,
        "spo2_flank_interval_sec": [0.3, 1.5],
        **settings,
    }
    provenance = {
        "algorithm_version": ALGORITHM_VERSION,
        "created_at_utc": utc_now(),
        "parameters": parameters,
        "channels": channel_info,
        "events": events,
        "paired_peak_protection_centres_sec": (paired / recording.fs_hz).tolist(),
        "sources": {
            "test_cleaning.py": BELT_SOURCE_SHA256,
            "test_spo2_cleaning.py": SPO2_SOURCE_SHA256,
        },
        "policy": "Automatic candidates are advisory; interpolation creates no manual review evidence",
    }
    identity = recording_identity(recording)
    signature = hashlib.sha256(
        json.dumps(
            {"identity": identity, "algorithm": ALGORITHM_VERSION, "parameters": parameters},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    candidates = candidate_intervals(recording, masks, events, signature, source_stage="early_cleaning")
    provenance["candidates"] = candidates
    corrected.setflags(write=False)
    for mask in masks.values():
        mask.setflags(write=False)
    return {
        "identity": identity,
        "signature": signature,
        "candidates": candidates,
        "corrected": corrected,
        "masks": masks,
        "provenance": provenance,
    }


def prepare_recording(recording, config):
    """Run automatic cleaning and source-oriented polarity assessment once.

    Returns corrected native data, automatic masks/events, a preparation
    signature and diagnostics. No manual evidence or polarity flip is inferred.
    """
    logger.info(
        "  Automatic correction and optional polarity assessment for %s...", recording.recording_id
    )
    prepared = clean_recording(recording, config)
    reliable = (
        np.isfinite(prepared["corrected"])
        & ~prepared["masks"]["interpolated"]
        & ~prepared["masks"]["auto_candidate"]
    )
    prepared["polarity"] = assess_relative_belt_polarity(
        prepared["corrected"],
        recording.channels,
        recording.fs_hz,
        reliable,
        config["preprocessing"]["check_relative_belt_polarity"],
    )
    logger.debug("Automatic cleaning: %s", prepared["provenance"]["channels"])
    logger.debug("Relative polarity: %s", prepared["polarity"])
    return prepared


def preprocess_ecg(values):
    """Pass through ECG; TODO: validated ECG-specific conditioning, beyond resampling."""
    return values.copy()


def preprocess_bp(values):
    """Pass through BP; TODO: validated BP-specific conditioning, beyond resampling."""
    return values.copy()
