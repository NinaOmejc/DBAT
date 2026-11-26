"""Moving-mean and SciPy Savitzky-Golay belt smoothing with separate QC support."""

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import savgol_filter

from ..config import window_samples
from .annotations import continuous_runs


def moving_mean(values, window):
    """Compute MATLAB-centered means with shrinking endpoints on a finite run.

    An even window uses window//2 earlier and window//2-1 later samples.
    Independently implemented using SciPy's uniform filter and support weights:
    https://www.mathworks.com/help/matlab/ref/movmean.html
    This matches the finite-input smoothdata(..., 'movmean', window) convention;
    no MATLAB numerical equivalence is claimed without a real MATLAB comparison.
    """
    means = uniform_filter1d(values, size=window, mode="constant", cval=0.0)
    weights = uniform_filter1d(np.ones(values.size), size=window, mode="constant", cval=0.0)
    return means / weights


def _window_influence(flagged, finite, window, method):
    """Map flagged input samples to actual moving-mean/SG calculation windows.

    Natural recording endpoints shrink (mean) or shift (SG) without inventing
    missing support. Internal gaps remain flagged through the nominal window.
    """
    influence = maximum_filter1d(flagged.astype(np.uint8), window, mode="constant", cval=0).astype(
        bool
    )
    if method == "savitzky_golay" and window > 1:
        margin = window // 2
        for start, end in continuous_runs(finite):
            if end - start >= window:
                influence[start : start + margin] |= flagged[start : start + window].any()
                influence[end - margin : end] |= flagged[end - window : end].any()
    return influence


def smoothing_masks(finite, support, eligible, interpolated, parameters):
    """Propagate support, QC eligibility and interpolation without smoothing values.

    Boolean vectors share the processed grid. Window/edge rules are identical
    to smooth_belt, including shrinking means and short Savitzky-Golay runs.
    """
    method, window = parameters["method"], parameters["window_samples"]
    if method == "none":
        return support.copy(), eligible & support, interpolated.copy()
    numeric_support = support.copy()
    if method == "savitzky_golay":
        for start, end in continuous_runs(finite):
            if end - start < window:
                numeric_support[start:end] = False
    numeric_support &= finite & ~_window_influence(~support, finite, window, method)
    valid = (
        numeric_support
        & eligible
        & ~_window_influence(~eligible, finite, window, method)
    )
    influence = _window_influence(interpolated, finite, window, method) & finite
    return numeric_support, valid, influence


def smooth_belt(values, support, eligible, interpolated, parameters):
    """Smooth finite processed runs once, retaining ordinary boundary support.

    Parameters
    ----------
    values : ndarray, shape (processed_samples,)
        Processed belt samples, never modified.
    support, eligible, interpolated : ndarray of bool
        Numerical support, QC eligibility and interpolation influence on this grid.
    parameters : dict
        Method, window_samples and optional Savitzky-Golay polyorder.

    Returns
    -------
    tuple
        Values, support, valid, interpolation influence and skipped SG runs.
        Moving means shrink even for short runs. Genuine gaps remain missing;
        neighbouring incomplete support is invalid without creating new NaNs.
    """
    finite = np.isfinite(values)
    method, window = parameters["method"], parameters["window_samples"]
    output = values.copy()
    skips = []
    if method == "none":
        return output, support.copy(), eligible & support, interpolated.copy(), skips
    for start, end in continuous_runs(finite):
        run = values[start:end]
        if method == "savitzky_golay":
            if len(run) < window:
                skips.append(
                    {
                        "start_sample": int(start),
                        "end_sample": int(end),
                        "reason": "shorter_than_savitzky_golay_window",
                    }
                )
                continue
            output[start:end] = savgol_filter(run, window, parameters["polyorder"], mode="interp")
        else:
            output[start:end] = moving_mean(run, window)
    numeric_support, valid, influence = smoothing_masks(
        finite, support, eligible, interpolated, parameters
    )
    return output, numeric_support, valid, influence, skips


def smooth_recording(
    preprocessed, recording, processed_fs, settings, masks, diagnostics,
    *, previous=None, changed=None
):
    """Smooth selected processed-grid belts once, updating masks and diagnostics.

    Parameters
    ----------
    preprocessed : ndarray, shape (processed_samples, channels)
        Native-conditioned, optionally resampled signals; left unchanged.
    recording : Recording
        Channel order; processed_fs is the actual smoothing rate in Hz.
    processed_fs : float
        Sampling frequency after optional primary resampling.
    settings : dict
        Existing preprocessing/smoothing settings.
    masks, diagnostics : dict
        Updated in place with smoothing support, influence and skipped runs.
    previous : PreprocessedRecording, optional
        Preview cache for unchanged channels; their masks still reflect current QC.
    changed : set of str, optional
        Channels whose numerical inputs changed, verified by the caller.

    Returns
    -------
    ndarray
        Smoothed values with the original shape, channel selection and timing.
    """
    processors = diagnostics["channel_processors"]
    processing_skips = diagnostics["processing_skips"]
    smoothing = settings["smoothing"]["method"] if settings["enabled"] else "none"
    smooth_window = window_samples(settings["smoothing"]["window_sec"], "seconds", processed_fs)
    if smoothing == "savitzky_golay":
        smooth_window = window_samples(
            settings["smoothing"]["sg_window"],
            settings["smoothing"]["sg_window_unit"],
            processed_fs,
        )
    smoothing_parameters = {
        "method": smoothing,
        "fs_hz": processed_fs,
        "window_samples": smooth_window if smoothing != "none" else None,
        "polyorder": settings["smoothing"]["polyorder"] if smoothing == "savitzky_golay" else None,
        "passes": 1 if smoothing != "none" else 0,
    }
    smoothed = preprocessed.copy()
    masks["smoothed_support"] = masks["preprocessed_support"].copy()
    masks["smoothed_interpolated"] = masks["preprocessed_interpolated"].copy()
    masks["valid"] = masks["preprocessed_valid"].copy()
    for column, name in enumerate(recording.channels):
        if processors[name] == "belt_conditioning":
            if previous is not None and name not in changed:
                smoothed[:, column] = previous.preproc_signal[:, column]
                (
                    masks["smoothed_support"][:, column],
                    masks["valid"][:, column],
                    masks["smoothed_interpolated"][:, column],
                ) = smoothing_masks(
                    np.isfinite(preprocessed[:, column]),
                    masks["preprocessed_support"][:, column],
                    masks["eligible"][:, column],
                    masks["preprocessed_interpolated"][:, column],
                    smoothing_parameters,
                )
                continue
            (
                smoothed[:, column],
                masks["smoothed_support"][:, column],
                masks["valid"][:, column],
                masks["smoothed_interpolated"][:, column],
                skips,
            ) = smooth_belt(
                preprocessed[:, column],
                masks["preprocessed_support"][:, column],
                masks["eligible"][:, column],
                masks["preprocessed_interpolated"][:, column],
                smoothing_parameters,
            )
            processing_skips[name]["smoothing_processed_grid"] = skips
    masks["finite"] = np.isfinite(preprocessed) & np.isfinite(smoothed)
    diagnostics["smoothing"] = smoothing_parameters
    return smoothed
