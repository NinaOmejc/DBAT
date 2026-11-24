"""Moving-mean and SciPy Savitzky-Golay belt smoothing with separate QC support."""

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import savgol_filter

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
    numeric_support = support.copy()
    for start, end in continuous_runs(finite):
        run = values[start:end]
        if method == "savitzky_golay":
            if len(run) < window:
                numeric_support[start:end] = False
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
    numeric_support &= finite & ~_window_influence(~support, finite, window, method)
    valid = numeric_support & eligible & ~_window_influence(~eligible, finite, window, method)
    influence = _window_influence(interpolated, finite, window, method) & finite
    return output, numeric_support, valid, influence, skips
