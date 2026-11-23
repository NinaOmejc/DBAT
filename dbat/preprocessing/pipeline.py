"""Condition reviewed continuous runs without changing supplied sample values."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from ..config import window_samples
from ..io.recording import Recording
from .annotations import annotation_masks, new_qc_annotations


@dataclass
class PreprocessingResult:
    """Native-grid derived views, independent masks and reproducible provenance."""

    recording: Recording
    conditioned: np.ndarray
    respiratory_view: np.ndarray
    masks: dict
    qc: dict
    provenance: dict


def continuous_runs(mask):
    """Return start/stop pairs for each true run in a 1-D native-grid mask."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2], strict=True)


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


def _condition_runs(values, eligible, method, window, polyorder=0):
    """Process eligible finite runs; mark incomplete window margins unsupported.

    Returns a derived vector and its full-window support mask. Short runs stay
    NaN. No interpolation or filtering across gaps is performed. Shrinking
    moving-mean edges are displayed but excluded from accepted analysis support.
    """
    output = np.full(values.shape, np.nan)
    supported = np.zeros(values.shape, dtype=bool)
    before, after = window // 2, (window - 1) // 2
    for start, end in continuous_runs(eligible):
        if end - start < window:
            continue
        run = values[start:end]
        if method == "savitzky_golay":
            result = savgol_filter(run, window, polyorder, mode="interp")
        else:
            trend = moving_mean(run, window)
            result = run - trend if method == "moving_detrend" else trend
            if method == "moving_detrend":
                # MATLAB detrend_flow removes the mean residual as well.
                # https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/detrend_flow.m
                result = result - np.mean(result)
        output[start:end] = result
        supported[start + before : end - after] = True
    return output, supported


def preprocess_recording(recording, config, qc_annotations=None):
    """Derive conditioning and respiratory views on the unchanged native grid.

    Parameters
    ----------
    recording : Recording
        Supplied float64 samples (n_samples, n_channels) and native fs in Hz.
    config : dict
        Validated grouped configuration. Detrending defaults to none.
    qc_annotations : dict, optional
        Accepted native QC; missing QC stays unreviewed, never implicitly clean.

    Returns
    -------
    PreprocessingResult
        Source, separate unsmoothed conditioned and smoothed respiratory views,
        finite/artifact/reviewed/uncertain/eligibility/support/valid masks.
        Actual conditioning needs reviewed coverage and an explicit usable rating.
    """
    if (
        recording.fs_hz != config["input"]["fs_hz"]
        or list(recording.channels) != config["input"]["channels"]
    ):
        raise ValueError("Recording fs/channel mapping disagrees with configuration")
    qc = qc_annotations if qc_annotations is not None else new_qc_annotations(recording)
    masks = annotation_masks(qc, recording)
    masks["finite"] = np.isfinite(recording.samples)
    usable = np.array([qc["channel_ratings"][name] == "usable" for name in recording.channels])
    unknown = np.array(
        [qc["channel_ratings"][name] in {"weak", "unassessed"} for name in recording.channels]
    )
    masks["uncertain"] = ~masks["reviewed"] | unknown[None, :]
    masks["eligible"] = masks["finite"] & masks["reviewed"] & ~masks["artifact"] & usable[None, :]
    conditioned = recording.samples.copy()
    conditioning_support = masks["finite"].copy()
    settings = config["preprocessing"]
    detrend = settings["detrend"]["method"] if settings["enabled"] else "none"
    smoothing = settings["smoothing"]["method"] if settings["enabled"] else "none"
    smooth_window = window_samples(settings["smoothing"]["window_sec"], "seconds", recording.fs_hz)
    if smoothing == "savitzky_golay":
        smooth_window = window_samples(
            settings["smoothing"]["sg_window"],
            settings["smoothing"]["sg_window_unit"],
            recording.fs_hz,
        )
    detrend_window = window_samples(settings["detrend"]["window_sec"], "seconds", recording.fs_hz)
    detrend_window += detrend_window % 2 == 0
    for column, name in enumerate(recording.channels):
        if name in settings["channels"] and detrend != "none":
            conditioned[:, column], conditioning_support[:, column] = _condition_runs(
                recording.samples[:, column], masks["eligible"][:, column], detrend, detrend_window
            )
    respiratory_view = conditioned.copy()
    respiratory_support = conditioning_support.copy()
    for column, name in enumerate(recording.channels):
        if name in settings["channels"] and smoothing != "none":
            respiratory_view[:, column], respiratory_support[:, column] = _condition_runs(
                conditioned[:, column],
                masks["eligible"][:, column] & conditioning_support[:, column],
                smoothing,
                smooth_window,
                settings["smoothing"]["polyorder"],
            )
    masks["conditioning_support"] = conditioning_support
    masks["respiratory_support"] = respiratory_support
    masks["conditioned_valid"] = masks["eligible"] & conditioning_support & np.isfinite(conditioned)
    masks["valid"] = masks["eligible"] & respiratory_support & np.isfinite(respiratory_view)
    provenance = {
        "algorithm_version": "dbat_conditioning_v1",
        "enabled": settings["enabled"],
        "selected_channels": settings["channels"],
        "detrend": {
            "method": detrend,
            "window_samples": detrend_window if detrend != "none" else None,
            "mean_offset": "per_continuous_run" if detrend != "none" else None,
        },
        "smoothing": {
            "method": smoothing,
            "window_samples": smooth_window if smoothing != "none" else None,
            "polyorder": settings["smoothing"]["polyorder"]
            if smoothing == "savitzky_golay"
            else None,
        },
        "edge_policy": "Run-local shrink (moving mean) or interp (SG); full-window margins excluded",
        "short_run_policy": "No window reduction; shorter than configured window => unsupported NaN view",
        "quality_policy": "Only explicitly inspected support on usable channels is conditioned/eligible",
        "polarity": "supplied_orientation_preserved; no manual or automatic polarity correction in Phase 1",
        "limitations": [
            "No MATLAB numerical comparison performed",
            "Review coverage may be partial",
            "Weak, unavailable and unassessed channels are ineligible",
            "No automatic artifact detector; uninspected samples remain unknown",
            "Respiratory smoothing is not a general spectral preprocessing stage",
        ],
        "counts": {
            name: {key: int(mask[:, column].sum()) for key, mask in masks.items()}
            for column, name in enumerate(recording.channels)
        },
    }
    return PreprocessingResult(recording, conditioned, respiratory_view, masks, qc, provenance)
