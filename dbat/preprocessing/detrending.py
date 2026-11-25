"""Native moving/robust-median detrending with independent support and diagnostics."""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d, median_filter, minimum_filter1d

from ..config import window_samples
from .annotations import continuous_runs
from .smoothing import moving_mean


ROBUST_MEDIAN_SOURCE_SHA256 = "f8fe03da99533265866ee6712ddd202eba25572d4bab643d9eb67e5902c986d5"


def robust_median_detrend(values, fs_hz, window_sec):
    """Return corrected values and native baseline, preserving its global median.

    Adapted exactly for finite runs from the user-supplied Sub100 experiment:
    Downloads/sub100_baseline_experiment/test_baseline_shift_sub100.py,
    robust_baseline_correction (source SHA-256 is recorded in provenance).
    One-second medians use complete blocks, a nearest-edge moving median, PCHIP
    at block centres (including tail extrapolation), then Gaussian sigma=0.7 s
    with reflect edges/truncate=4. No selective abrupt-step correction is used.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html

    Parameters
    ----------
    values : ndarray, shape (native_samples,)
        One finite non-artifact run containing at least two complete blocks.
    fs_hz, window_sec : float
        Native sampling rate in Hz and median window duration in seconds.
    """
    block = window_samples(1.0, "seconds", fs_hz)
    whole = len(values) // block
    if whole < 2:
        raise ValueError("Robust median detrending needs two complete one-second blocks")
    coarse = np.median(values[: whole * block].reshape(whole, block), axis=1)
    window = window_samples(window_sec, "seconds", fs_hz / block)
    baseline = median_filter(coarse, size=window, mode="nearest")
    baseline_native = PchipInterpolator(
        (np.arange(whole) + 0.5) * block / fs_hz,
        baseline,
        extrapolate=True,
    )(np.arange(len(values)) / fs_hz)
    baseline_native = gaussian_filter1d(
        baseline_native, sigma=0.7 * fs_hz, mode="reflect", truncate=4.0
    )
    return values - baseline_native + np.median(baseline_native), baseline_native


def detrending_parameters(settings, fs_hz):
    """Resolve independent moving-mean/robust-median settings and provenance."""
    method = settings["detrend"]["method"] if settings["enabled"] else "none"
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
        seconds = settings["detrend"]["robust_median_window_sec"]
        block = window_samples(1.0, "seconds", fs_hz)
        coarse_window = window_samples(seconds, "seconds", fs_hz / block)
        parameters.update(
            window_sec=seconds,
            block_samples=block,
            effective_block_sec=block / fs_hz,
            median_window_blocks=coarse_window,
            effective_window_sec=coarse_window * block / fs_hz,
            median_edges="nearest",
            interpolation="scipy.interpolate.PchipInterpolator",
            partial_block="excluded from medians; baseline extrapolated to original tail",
            gaussian_sigma_sec=0.7,
            gaussian_edges="reflect",
            gaussian_truncate=4.0,
            mean_offset="median_of_estimated_baseline_per_run",
            support_radius_samples=(coarse_window // 2 + 6) * block,
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
        Method and effective moving-mean or robust-median parameters on the native grid.

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
            if end - start < window:
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
                output[start:end], _ = robust_median_detrend(
                    values[start:end], detrend["fs_hz"], detrend["window_sec"]
                )
            else:
                # https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/detrend_flow.m
                residual = values[start:end] - moving_mean(values[start:end], window)
                output[start:end] = residual - np.mean(residual)
            if influence[start:end].any():
                influence[start:end] = True  # Run-wide retained baseline/mean offset.
        support_window = 2 * detrend["support_radius_samples"] + 1 if robust else window
        support &= minimum_filter1d(
            allowed.astype(np.uint8), support_window, mode="constant", cval=1
        ).astype(bool)
    return output, allowed, support, influence, skips
