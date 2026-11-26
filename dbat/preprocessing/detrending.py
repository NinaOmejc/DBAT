"""Native moving/robust-median detrending with independent support and diagnostics."""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d, median_filter, minimum_filter1d

from ..config import window_samples
from .annotations import continuous_runs
from .smoothing import moving_mean


ROBUST_MEDIAN_SOURCE_SHA256 = "f8fe03da99533265866ee6712ddd202eba25572d4bab643d9eb67e5902c986d5"


def robust_median_detrend(values, fs_hz, window_sec, transition_sec=25.0):
    """Return a robust baseline correction with unchanged half-window edges.

    Reuses the user-supplied Sub100 experiment's one-second medians, PCHIP and
    Gaussian sigma=0.7 s (source hash below). Unlike that experiment, only fully
    supported median windows enter interpolation. The correction fades with a
    raised cosine after each half-window guard; overlapping fades shorten to
    meet at the segment midpoint. Too-short finite segments remain unchanged.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html

    Parameters
    ----------
    values : ndarray, shape (native_samples,)
        One finite native segment; not modified.
    fs_hz : float
        Native sampling rate in Hz.
    window_sec, transition_sec : float
        Median window and maximum fade durations in seconds.

    Returns
    -------
    tuple of ndarray
        Corrected segment and estimated baseline; baseline is NaN for skipped segments.
    """
    block = window_samples(1.0, "seconds", fs_hz)
    whole = len(values) // block
    window = window_samples(window_sec, "seconds", fs_hz / block)
    left, right = window // 2, (window - 1) // 2
    times = np.arange(len(values)) / fs_hz
    distance = np.minimum(times, times[-1] - times)
    fade_sec = min(transition_sec, max(0.0, times[-1] / 2 - window_sec / 2))
    if whole < window + 1 or fade_sec <= 0:
        return values.copy(), np.full(len(values), np.nan)
    coarse = np.median(values[: whole * block].reshape(whole, block), axis=1)
    centres = (np.arange(whole) + 0.5) * block / fs_hz
    # Padding values are discarded: interpolation sees complete windows only.
    baseline = median_filter(coarse, size=window, mode="reflect")[left : whole - right]
    centres = centres[left : whole - right]
    baseline_native = PchipInterpolator(centres, baseline, extrapolate=False)(
        np.clip(times, centres[0], centres[-1])
    )
    baseline_native = gaussian_filter1d(
        baseline_native, sigma=0.7 * fs_hz, mode="reflect", truncate=4.0
    )
    phase = np.clip((distance - window_sec / 2) / fade_sec, 0.0, 1.0)
    weight = 0.5 - 0.5 * np.cos(np.pi * phase)
    output = values.copy()
    active = weight > 0
    output[active] -= weight[active] * (
        baseline_native[active] - np.median(baseline_native)
    )
    return output, baseline_native


def detrending_parameters(settings, fs_hz, channel=None, qc=None):
    """Resolve native parameters: QC belt choice, config belt default, then method.

    An explicit On selects robust_median; Off selects none. Older configs/QC
    without per-belt choices retain their method setting. Global enablement and
    selected channels gate numerical processing; runs is filled by detrend_belt.
    """
    options = settings["detrend"]
    decisions = (qc or {}).get("baseline_correction", {})
    defaults = options.get("baseline_correction", {})
    method = options["method"]
    if channel in decisions or channel in defaults:
        enabled = decisions[channel] if channel in decisions else defaults[channel]
        method = "robust_median" if enabled else "none"
    if not settings["enabled"] or (channel and channel not in settings["channels"]):
        method = "none"
    seconds = settings["detrend"]["window_sec"]
    window = window_samples(seconds, "seconds", fs_hz)
    window += window % 2 == 0
    parameters = {
        "method": method,
        "fs_hz": fs_hz,
        "window_samples": window if method == "moving_detrend" else None,
        "mean_offset": "per_finite_non_artifact_run" if method == "moving_detrend" else None,
    }
    if method == "robust_median":
        block = window_samples(1.0, "seconds", fs_hz)
        coarse_window = window_samples(seconds, "seconds", fs_hz / block)
        parameters.update(
            window_sec=seconds,
            block_samples=block,
            effective_block_sec=block / fs_hz,
            median_window_blocks=coarse_window,
            effective_window_sec=coarse_window * block / fs_hz,
            median_edges="fully_supported_windows_only",
            edge_mode="skip_half_window_then_raised_cosine_fade",
            skip_each_end_sec=seconds / 2,
            transition_sec=settings["detrend"].get("transition_sec", 25.0),
            short_segment_policy="unchanged when no fully supported baseline or room for fades",
            runs=[],
            interpolation="scipy.interpolate.PchipInterpolator",
            partial_block="excluded from medians; half-window tail remains unchanged",
            gaussian_sigma_sec=0.7,
            gaussian_edges="reflect",
            gaussian_truncate=4.0,
            mean_offset="median_of_estimated_baseline_per_run",
            source_experiment="test_baseline_shift_sub100.py::robust_baseline_correction",
            source_sha256=ROBUST_MEDIAN_SOURCE_SHA256,
        )
    return parameters


def detrend_belt(values, artifact, interpolated, detrend):
    """Detrend a native belt, retaining the original filter-eligibility mask.

    Parameters
    ----------
    values : ndarray, shape (native_samples,)
        Corrected, polarity-oriented samples; never modified.
    artifact, interpolated : ndarray of bool
        Confirmed artifacts and existing interpolation influence.
    detrend : dict
        Effective native parameters; robust per-segment edge provenance is added in place.

    Returns
    -------
    tuple
        Values, original allowed mask, detrending support, interpolation
        influence and skipped native runs. None of these masks create new NaNs.
    """
    allowed = np.isfinite(values) & ~artifact
    output, support, influence = values.copy(), allowed.copy(), interpolated.copy()
    skips = []
    if detrend["method"] != "none":
        robust = detrend["method"] == "robust_median"
        window = 2 * detrend["block_samples"] if robust else detrend["window_samples"]
        for start, end in continuous_runs(allowed):
            if end - start < window and not robust:
                support[start:end] = False
                skips.append(
                    {
                        "start_sample": int(start),
                        "end_sample": int(end),
                        "reason": "shorter_than_detrend_window",
                    }
                )
                continue
            if robust:
                output[start:end], baseline = robust_median_detrend(
                    values[start:end],
                    detrend["fs_hz"],
                    detrend["window_sec"],
                    detrend["transition_sec"],
                )
                applied = bool(np.isfinite(baseline).any())
                edge = {
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "applied": applied,
                    "skip_each_end_sec": detrend["skip_each_end_sec"],
                    "effective_transition_sec": min(
                        detrend["transition_sec"],
                        max(
                            0.0,
                            (end - start - 1) / (2 * detrend["fs_hz"])
                            - detrend["skip_each_end_sec"],
                        ),
                    ) if applied else 0.0,
                }
                if not applied:
                    edge["reason"] = "insufficient_segment_for_guarded_baseline"
                    skips.append(dict(edge))
                detrend["runs"].append(edge)
            else:
                # https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/detrend_flow.m
                residual = values[start:end] - moving_mean(values[start:end], window)
                output[start:end] = residual - np.mean(residual)
            if influence[start:end].any():
                if robust and applied:
                    guard = detrend["skip_each_end_sec"] * detrend["fs_hz"]
                    distance = np.minimum(np.arange(end - start), np.arange(end - start)[::-1])
                    influence[start:end] |= distance > guard
                elif not robust:
                    influence[start:end] = True  # Run-wide retained mean offset.
        if not robust:
            support &= minimum_filter1d(
                allowed.astype(np.uint8), window, mode="constant", cval=1
            ).astype(bool)
    return output, allowed, support, influence, skips
