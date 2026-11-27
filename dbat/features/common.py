"""Shared QC, interval support and native-resolution result construction."""

import numpy as np
from scipy import integrate

from ..preprocessing.annotations import continuous_runs


def observations(values, time, *, start=None, end=None, coverage=None, **metadata):
    """Package 1-D observations with explicit seconds, missingness and support."""
    values, time = np.asarray(values, float), np.asarray(time, float)
    if values.shape != time.shape or values.ndim != 1:
        raise ValueError("Feature values and timestamps must be aligned vectors")
    valid = np.isfinite(values)
    result = dict(
        values=np.where(valid, values, np.nan),
        time_sec=time,
        valid=valid,
        available=bool(valid.any()),
        quality="good"
        if valid.all() and len(valid)
        else ("partial" if valid.any() else "unavailable"),
        **metadata,
    )
    result["support_start_sec"] = np.asarray(time if start is None else start, float)
    result["support_end_sec"] = np.asarray(time if end is None else end, float)
    if coverage is not None:
        result["coverage"] = np.asarray(coverage, float)
    return result


def channel_signal(data, channel):
    """Return final processed samples and current QC eligibility; warnings are advisory."""
    column = data["channels"].index(channel)
    values, masks = data["preproc_signal"][:, column], data["masks"]
    valid = (
        masks["valid"][:, column]
        & ~masks["artifact"][:, column]
        & ~masks["uncertain"][:, column]
        & np.isfinite(values)
    )
    if data["metadata"]["qc"]["channel_ratings"].get(channel) != "usable":
        valid = np.zeros(len(values), bool)
    return np.where(valid, values, np.nan), valid


def reference_value(reference, key="value"):
    """Read one available positive respiratory reference; never substitute another."""
    value = reference.get(key)
    return (
        float(value)
        if reference.get("available")
        and value is not None
        and np.isfinite(value)
        and value > 0
        else np.nan
    )


def normalized_belt(values):
    """MAGMA-ml median/IQR normalization, with population-SD fallback for flat IQR."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.full(len(values), np.nan)
    scale = np.percentile(finite, 75) - np.percentile(finite, 25)
    if scale <= np.finfo(float).eps:
        scale = np.std(finite)
    if scale <= np.finfo(float).eps:
        return np.full(len(values), np.nan)
    return (values - np.median(finite)) / scale


def trailing_support(valid, fs, step, window):
    """Yield complete MATLAB-style trailing supports, including the endpoint sample."""
    grid = (
        np.arange(0, (len(valid) - 1) / fs + 1e-9, step) if len(valid) else np.array([])
    )
    bad = np.r_[0, np.cumsum(~valid)]
    for t in grid:
        first = max(0, int(np.floor((t - window) * fs)))
        stop = min(len(valid), int(np.floor(t * fs)) + 1)
        coverage = (
            1 - (bad[stop] - bad[first]) / (stop - first) if stop > first else 0.0
        )
        yield t, first, stop, t >= window and coverage == 1.0, coverage


def spectral_windows(valid, fs, window, step):
    """Yield fixed half-open windows wholly within each independent valid run."""
    count, stride = (
        max(2, int(np.floor(window * fs + 0.5))),
        max(1, int(np.floor(step * fs + 0.5))),
    )
    for first, stop in continuous_runs(valid):
        for start in range(first, stop - count + 1, stride):
            yield start, start + count


def band_power(frequency, power, band):
    """Integrate linearly interpolated band edges, as in MAGMA-ml respiratory.py."""
    low, high = max(frequency[0], band[0]), min(frequency[-1], band[1])
    if high <= low:
        return np.nan
    points = np.r_[low, frequency[(frequency > low) & (frequency < high)], high]
    return float(integrate.trapezoid(np.interp(points, frequency, power), points))
