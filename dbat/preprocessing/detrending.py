"""Native moving detrending, support masks and skipped-run diagnostics."""

import numpy as np
from scipy.ndimage import minimum_filter1d

from ..config import window_samples
from .annotations import continuous_runs
from .smoothing import moving_mean


def detrending_parameters(settings, fs_hz):
    """Resolve native detrending seconds into the existing odd sample window.

    Returns the unchanged method/fs/window/mean-offset provenance dictionary.
    """
    detrend_window = window_samples(settings["detrend"]["window_sec"], "seconds", fs_hz)
    detrend_window += detrend_window % 2 == 0
    detrend = settings["detrend"]["method"] if settings["enabled"] else "none"
    detrend_parameters = {
        "method": detrend,
        "fs_hz": fs_hz,
        "window_samples": detrend_window if detrend != "none" else None,
        "mean_offset": "per_finite_non_artifact_run" if detrend != "none" else None,
    }
    return detrend_parameters


def detrend_belt(values, artifact, interpolated, detrend):
    """Detrend a native belt, retaining the original filter-eligibility mask.

    Parameters
    ----------
    values : ndarray, shape (native_samples,)
        Corrected, polarity-oriented samples; never modified.
    artifact, interpolated : ndarray of bool
        Confirmed artifacts and existing interpolation influence.
    detrend : dict
        Method and effective odd window_samples on the native grid.

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
        window = detrend["window_samples"]
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
            # Adapted from the MIT MATLAB detrend_flow arithmetic:
            # https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/detrend_flow.m
            residual = values[start:end] - moving_mean(values[start:end], window)
            output[start:end] = residual - np.mean(residual)
            if influence[start:end].any():
                influence[start:end] = True  # Run-wide residual-mean removal.
        support &= minimum_filter1d(
            allowed.astype(np.uint8), window, mode="constant", cval=1
        ).astype(bool)
    return output, allowed, support, influence, skips
