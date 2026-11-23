"""Condition reviewed continuous runs without changing supplied sample values."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from ..config import BELTS, window_samples
from ..io.recording import Recording
from .annotations import annotation_masks, new_qc_annotations
from .resampling import resample_views


@dataclass
class PreprocessingResult:
    """Native views/QC, optional common-grid resampling and reproducible provenance."""

    recording: Recording
    conditioned: np.ndarray
    respiratory_view: np.ndarray
    masks: dict
    qc: dict
    provenance: dict
    resampled: dict | None = None


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


def _passthrough_channel(values):
    """Return separate unchanged views and finite-only numerical support."""
    finite = np.isfinite(values)
    return values.copy(), values.copy(), finite, finite.copy()


def preprocess_belt(values, eligible, detrend, smoothing):
    """Condition a native belt vector within reviewed, usable finite runs.

    Parameters
    ----------
    values : ndarray, shape (n_samples,)
        Native belt samples; never modified.
    eligible : ndarray of bool
        Finite, reviewed, non-artifact support on a usable channel.
    detrend, smoothing : dict
        Methods and effective native sample windows.

    Returns
    -------
    tuple of ndarray
        Unsmoothed/smoothed vectors, then their full-window support masks.
        Processing stays within eligible runs; gaps are never bridged.
    """
    conditioned = values.copy()
    conditioning_support = np.isfinite(values)
    if detrend["method"] != "none":
        conditioned, conditioning_support = _condition_runs(
            values, eligible, detrend["method"], detrend["window_samples"]
        )
    view, view_support = conditioned.copy(), conditioning_support.copy()
    if smoothing["method"] != "none":
        view, view_support = _condition_runs(
            conditioned,
            eligible & conditioning_support,
            smoothing["method"],
            smoothing["window_samples"],
            smoothing["polyorder"] or 0,
        )
    return conditioned, view, conditioning_support, view_support


def preprocess_spo2(values, eligible):
    """Prepare native SpO2 using finite/manual QC, preserving the supplied scale.

    Parameters
    ----------
    values : ndarray, shape (n_samples,)
        Native SpO2 samples in their supplied units.
    eligible : ndarray of bool
        Reviewed, non-artifact support on a usable channel.

    Returns
    -------
    tuple of ndarray
        Two independent QC-masked views, then their finite/eligible masks.
        No filtering, rescaling, clipping or physiological thresholds are applied.
    """
    supported = np.isfinite(values) & eligible
    view = np.where(supported, values, np.nan)
    return view, view.copy(), supported, supported.copy()


def preprocess_ecg(values):
    """Pass through native ECG; TODO: add validated ECG-specific conditioning.

    Returns two independent views and finite support; shared resampling is applied
    afterwards by preprocess_recording, independently of this placeholder.
    """
    return _passthrough_channel(values)


def preprocess_bp(values):
    """Pass through native BP; TODO: add validated BP-specific conditioning.

    Returns two independent views and finite support. Common resampling remains
    available without implying that blood-pressure conditioning is implemented.
    """
    return _passthrough_channel(values)


def preprocess_recording(recording, config, qc_annotations=None):
    """Prepare signal-specific native views and optionally resample all channels.

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
        Quality approval requires selection in qc.channels, reviewed coverage and
        an explicit usable rating. Optional resampled views have a separate grid.
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
    selected = np.array([name in config["qc"]["channels"] for name in recording.channels])
    masks["quality_selected"] = np.broadcast_to(selected, recording.samples.shape).copy()
    masks["uncertain"] = ~masks["reviewed"] | unknown[None, :] | ~masks["quality_selected"]
    masks["eligible"] = (
        masks["quality_selected"] & masks["finite"] & masks["reviewed"]
        & ~masks["artifact"] & usable[None, :]
    )
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
    detrend_parameters = {
        "method": detrend,
        "window_samples": detrend_window if detrend != "none" else None,
        "mean_offset": "per_continuous_run" if detrend != "none" else None,
    }
    smoothing_parameters = {
        "method": smoothing,
        "window_samples": smooth_window if smoothing != "none" else None,
        "polyorder": settings["smoothing"]["polyorder"] if smoothing == "savitzky_golay" else None,
    }
    conditioned = np.empty_like(recording.samples)
    respiratory_view = np.empty_like(recording.samples)
    conditioning_support = np.empty_like(masks["finite"])
    respiratory_support = np.empty_like(masks["finite"])
    processors = {}
    for column, name in enumerate(recording.channels):
        values, eligible = recording.samples[:, column], masks["eligible"][:, column]
        if name in BELTS and name in settings["channels"] and selected[column]:
            channel_result = preprocess_belt(values, eligible, detrend_parameters, smoothing_parameters)
            processors[name] = "belt_conditioning"
        elif name == "SpO2" and selected[column]:
            channel_result = preprocess_spo2(values, eligible)
            processors[name] = "spo2_manual_qc_only"
        elif name in {"ECG1", "ECG2"}:
            channel_result = preprocess_ecg(values)
            processors[name] = "ecg_todo_passthrough"
        elif name == "Blood Pressure":
            channel_result = preprocess_bp(values)
            processors[name] = "bp_todo_passthrough"
        else:
            channel_result = _passthrough_channel(values)
            processors[name] = "unselected_passthrough"
        (
            conditioned[:, column],
            respiratory_view[:, column],
            conditioning_support[:, column],
            respiratory_support[:, column],
        ) = channel_result
    masks["conditioning_support"] = conditioning_support
    masks["respiratory_support"] = respiratory_support
    masks["conditioned_valid"] = masks["eligible"] & conditioning_support & np.isfinite(conditioned)
    masks["valid"] = masks["eligible"] & respiratory_support & np.isfinite(respiratory_view)
    provenance = {
        "algorithm_version": "dbat_conditioning_v2",
        "enabled": settings["enabled"],
        "selected_channels": settings["channels"],
        "qc_channels": config["qc"]["channels"],
        "channel_processors": processors,
        "detrend": detrend_parameters,
        "smoothing": smoothing_parameters,
        "edge_policy": "Run-local shrink (moving mean) or interp (SG); full-window margins excluded",
        "short_run_policy": "No window reduction; shorter than configured window => unsupported NaN view",
        "quality_policy": "Only selected QC channels with explicitly inspected support and usable ratings are eligible",
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
    result = PreprocessingResult(recording, conditioned, respiratory_view, masks, qc, provenance)
    result.resampled, provenance["resampling"] = resample_views(result, settings["resampling"])
    return result
