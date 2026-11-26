"""Finite-pair MAE measures preprocessing impact, never signal quality or validity."""

from copy import deepcopy

import numpy as np

from ..config import BELTS, window_samples


def preprocessing_impact(
    recording, original, cleaned, detrended, final, processed_fs_hz,
    *, previous=None, changed=None
):
    """Calculate four stage MAEs in 30 s/10 s windows and over each whole belt.

    Native stages share the applied polarity. Native-to-final comparisons use
    linear interpolation onto final timestamps only where both bracketing native
    samples are finite and timestamps overlap. No filtering of the reference or
    missing-data bridging is used. Coverage <90% produces None. Recording MAE
    uses unique sample pairs, not double-counted overlapping windows.

    Parameters
    ----------
    recording : Recording
        Native channel order, zero-origin time axis and sampling rate in Hz.
    original, cleaned, detrended : ndarray, shape (native_samples, channels)
        Native stage values in the same polarity, read without modification.
    final : ndarray, shape (processed_samples, channels)
        Final waveform on the zero-origin processed grid.
    processed_fs_hz : float
        Processed sampling rate in Hz.
    previous : dict, optional
        Cached impact metrics; unchanged channels are copied without recalculation.
    changed : set of str, optional
        Channels with changed numerical inputs, checked by the caller.

    Returns
    -------
    dict
        Serializable window/recording MAEs, finite-pair coverage and settings.
    """
    window, step = (
        window_samples(value, "seconds", recording.fs_hz) for value in (30.0, 10.0)
    )
    count = len(original)
    starts = list(range(0, max(0, count - window + 1), step)) or [0]
    times = np.arange(len(final)) / processed_fs_hz
    position = times * recording.fs_hz
    position = np.where(
        np.isclose(position, np.rint(position), atol=1e-9, rtol=0),
        np.rint(position),
        position,
    )
    left, right = np.floor(position).astype(int), np.ceil(position).astype(int)
    in_range = (times >= recording.time_sec[0]) & (times <= recording.time_sec[-1])
    left, right = np.clip(left, 0, count - 1), np.clip(right, 0, count - 1)
    result = {
        "algorithm_version": "dbat_preprocessing_impact_v1",
        "window_sec": 30.0,
        "step_sec": 10.0,
        "min_coverage_fraction": 0.9,
        "processed_fs_hz": processed_fs_hz,
        "polarity_policy": "All comparisons use the same applied source-relative polarity; sign flips themselves are excluded",
        "grid_policy": "Native for first two comparisons; finite-bracket linear alignment to final timestamps for last two",
        "unassessed_tail_sec": (count - min(count, starts[-1] + window))
        / recording.fs_hz,
        "window_policy": "Complete 30-second windows; a short-recording window is explicitly undefined",
        "policy": "Impact only; never used to flag or exclude data; no intermediate waveforms are saved",
        "channels": {},
    }
    for column, channel in enumerate(recording.channels):
        if channel not in BELTS:
            continue
        if previous is not None and channel not in changed:
            result["channels"][channel] = deepcopy(previous["channels"][channel])
            continue

        def aligned(values):
            vector = values[:, column]
            if processed_fs_hz == recording.fs_hz and len(final) == count:
                return vector
            valid = in_range & np.isfinite(vector[left]) & np.isfinite(vector[right])
            output = np.full(len(final), np.nan)
            fraction = position[valid] - left[valid]
            output[valid] = (
                vector[left[valid]] * (1 - fraction) + vector[right[valid]] * fraction
            )
            return output

        pairs = {
            "original_to_cleaned": (
                original[:, column],
                cleaned[:, column],
                recording.time_sec,
            ),
            "cleaned_to_detrended": (
                cleaned[:, column],
                detrended[:, column],
                recording.time_sec,
            ),
            "detrended_to_final": (aligned(detrended), final[:, column], times),
            "original_to_final": (aligned(original), final[:, column], times),
        }

        def measure(first, last, *, full=True):
            values, coverage, samples = {}, {}, {}
            for name, (before, after, axis) in pairs.items():
                lo, hi = np.searchsorted(axis, [first, last])
                before, after = before[lo:hi], after[lo:hi]
                finite = np.isfinite(before) & np.isfinite(after)
                n, total = int(finite.sum()), hi - lo
                fraction = n / total if total else 0.0
                coverage[name], samples[name] = fraction, n
                values[name] = (
                    float(np.mean(np.abs(before[finite] - after[finite])))
                    if full and n and fraction >= 0.9
                    else None
                )
            return {
                "mae": values,
                "coverage_fraction": coverage,
                "paired_samples": samples,
            }

        rows = [
            {
                "start_sec": start / recording.fs_hz,
                "end_sec": min(count, start + window) / recording.fs_hz,
                "status": "assessed"
                if start + window <= count
                else "insufficient_duration",
                **measure(
                    start / recording.fs_hz,
                    min(count, start + window) / recording.fs_hz,
                    full=start + window <= count,
                ),
            }
            for start in starts
        ]
        result["channels"][channel] = {
            "recording": measure(0.0, count / recording.fs_hz),
            "windows": rows,
        }
    return result
