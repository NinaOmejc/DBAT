"""Build numerical preprocessed/smoothed signals with independent analysis masks."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import savgol_filter

from ..config import BELTS, window_samples
from ..io.recording import Recording
from ..logging import logger
from .annotations import annotation_masks, continuous_runs, new_qc_annotations, recording_identity
from .cleaning import clean_recording, preprocess_spo2 as preprocess_spo2
from .polarity import apply_polarity, assess_relative_belt_polarity, saved_polarity_decision
from .resampling import resample_views


@dataclass
class PreprocessingResult:
    """Original, preprocessed and authoritative smoothed signals with separate QC."""

    recording: Recording
    preprocessed: np.ndarray
    smoothed: np.ndarray
    masks: dict
    qc: dict
    provenance: dict
    resampled: dict | None = None

    @property
    def original(self):
        """Return the immutable supplied native samples without making a copy."""
        return self.recording.samples

    @property
    def respiratory_signal(self):
        """Return smoothed, the sole respiratory-analysis input; obey masks['valid']."""
        return self.smoothed


def prepare_recording(recording, config):
    """Run automatic cleaning and source-oriented polarity assessment once.

    Returns corrected native data, automatic masks/events, a compatibility
    signature and diagnostics. No manual evidence or polarity flip is inferred.
    """
    logger.info("  Automatic correction and optional polarity assessment for %s...", recording.recording_id)
    prepared = clean_recording(recording, config)
    reliable = (
        np.isfinite(prepared["corrected"])
        & ~prepared["masks"]["interpolated"] & ~prepared["masks"]["auto_candidate"]
    )
    prepared["polarity"] = assess_relative_belt_polarity(
        prepared["corrected"], recording.channels, recording.fs_hz, reliable,
        config["preprocessing"]["check_relative_belt_polarity"],
    )
    logger.debug("Automatic cleaning: %s", prepared["provenance"]["channels"])
    logger.debug("Relative polarity: %s", prepared["polarity"])
    return prepared


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

    Returns numerical values and full-window support independently. Excluded or
    short runs retain their input values with false support. Source nonfinite
    values remain nonfinite. Filtering never crosses a gap or QC exclusion;
    calculated moving-mean/SG boundary values have false full-window support.
    """
    output = values.copy()
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
    preprocessed = values.copy()
    preprocessed_support = np.isfinite(values)
    if detrend["method"] != "none":
        preprocessed, preprocessed_support = _condition_runs(
            values, eligible, detrend["method"], detrend["window_samples"]
        )
    smoothed, smoothed_support = preprocessed.copy(), preprocessed_support.copy()
    if smoothing["method"] != "none":
        smoothed, smoothed_support = _condition_runs(
            preprocessed,
            eligible & preprocessed_support,
            smoothing["method"],
            smoothing["window_samples"],
            smoothing["polyorder"] or 0,
        )
    return preprocessed, smoothed, preprocessed_support, smoothed_support



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


def preprocess_recording(recording, config, qc_annotations=None, prepared=None):
    """Prepare signal-specific native views and optionally resample all channels.

    Parameters
    ----------
    recording : Recording
        Supplied float64 samples (n_samples, n_channels) and native fs in Hz.
    config : dict
        Validated grouped configuration. Detrending defaults to none.
    qc_annotations : dict, optional
        Accepted native QC; missing QC stays unreviewed, never implicitly clean.
    prepared : dict, optional
        Cached prepare_recording result, shared with the initial QC GUI.

    Returns
    -------
    PreprocessingResult
        Immutable original, preprocessed and authoritative smoothed waveforms,
        finite/artifact/reviewed/uncertain/eligibility/support/valid masks.
        Quality approval requires selection in qc.channels, reviewed coverage and
        an explicit usable rating. Optional resampled views have a separate grid.
    """
    if (
        recording.fs_hz != config["input"]["fs_hz"]
        or list(recording.channels) != config["input"]["channels"]
    ):
        raise ValueError("Recording fs/channel mapping disagrees with configuration")
    logger.info("  Preprocessing %s...", recording.recording_id)
    qc = qc_annotations if qc_annotations is not None else new_qc_annotations(recording)
    masks = annotation_masks(qc, recording)
    prepared = prepare_recording(recording, config) if prepared is None else prepared
    if prepared["identity"] != recording_identity(recording) or prepared["corrected"].shape != recording.samples.shape:
        raise ValueError("Prepared signals do not match the source identity/native grid")
    decision = saved_polarity_decision(qc, prepared)
    choice = decision["choice"] if decision else None
    if prepared["polarity"]["polarity_warning"] and choice is None and not (decision and decision["acknowledged_warning"]):
        raise ValueError("Polarity warning requires an explicit reviewer choice or acknowledgement in the QC GUI")
    corrected = apply_polarity(prepared["corrected"], recording.channels, choice)
    masks.update(prepared["masks"])
    masks["finite"] = np.isfinite(corrected)
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
    preprocessed = np.empty_like(corrected)
    smoothed = np.empty_like(corrected)
    preprocessed_support = np.empty_like(masks["finite"])
    smoothed_support = np.empty_like(masks["finite"])
    processors = {}
    processing_skips = {}
    masks["preprocessed_interpolated"] = masks["interpolated"].copy()
    masks["smoothed_interpolated"] = masks["interpolated"].copy()
    for column, name in enumerate(recording.channels):
        values, eligible = corrected[:, column], masks["eligible"][:, column]
        if name in BELTS and name in settings["channels"] and selected[column]:
            channel_result = preprocess_belt(values, eligible, detrend_parameters, smoothing_parameters)
            processors[name] = "belt_conditioning"
        elif name == "SpO2":
            channel_result = _passthrough_channel(values)
            processors[name] = "spo2_corrected_without_filtering"
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
            preprocessed[:, column],
            smoothed[:, column],
            preprocessed_support[:, column],
            smoothed_support[:, column],
        ) = channel_result
        if name in BELTS and name in settings["channels"] and selected[column]:
            smooth_eligible = eligible & preprocessed_support[:, column]
            processing_skips[name] = {}
            for stage, method, allowed, window in (
                ("preprocessed", detrend, eligible, detrend_window),
                ("smoothed", smoothing, smooth_eligible, smooth_window),
            ):
                if method != "none":
                    processing_skips[name][stage] = {
                        "excluded_samples_preserved": int((~allowed).sum()),
                        "short_runs_preserved": [
                            {"start_sample": int(start), "end_sample": int(end),
                             "reason": "shorter_than_configured_window"}
                            for start, end in continuous_runs(allowed) if end - start < window
                        ],
                    }
            dependence = masks["interpolated"][:, column].copy()
            if detrend != "none":
                for start, end in continuous_runs(eligible):
                    if end - start >= detrend_window and dependence[start:end].any():
                        dependence[start:end] = True  # Run-wide residual-mean removal.
            masks["preprocessed_interpolated"][:, column] = dependence
            smoothed_dependence = dependence.copy()
            if smoothing != "none":
                for start, end in continuous_runs(smooth_eligible):
                    if end - start < smooth_window:
                        continue
                    run = dependence[start:end]
                    influence = maximum_filter1d(
                        run.astype(np.uint8), smooth_window, mode="constant", cval=0,
                    ).astype(bool)
                    if smoothing == "savitzky_golay" and smooth_window > 1:
                        margin = smooth_window // 2
                        influence[:margin] = run[:smooth_window].any()
                        influence[-margin:] = run[-smooth_window:].any()
                    smoothed_dependence[start:end] = influence
            masks["smoothed_interpolated"][:, column] = smoothed_dependence
    masks["preprocessed_interpolated"] &= np.isfinite(preprocessed)
    masks["smoothed_interpolated"] &= np.isfinite(smoothed)
    masks["preprocessed_support"] = preprocessed_support
    masks["smoothed_support"] = smoothed_support
    masks["preprocessed_valid"] = masks["eligible"] & preprocessed_support & np.isfinite(preprocessed)
    masks["valid"] = masks["eligible"] & smoothed_support & np.isfinite(smoothed)
    provenance = {
        "algorithm_version": "dbat_preprocessing_v4",
        "enabled": settings["enabled"],
        "selected_channels": settings["channels"],
        "qc_channels": config["qc"]["channels"],
        "channel_processors": processors,
        "detrend": detrend_parameters,
        "smoothing": smoothing_parameters,
        "edge_policy": "Run-local shrink (moving mean) or interp (SG); full-window margins excluded",
        "short_run_policy": "Keep input values; support false when the configured window cannot be applied",
        "signal_policy": "Numerical values are independent of QC and processing-support masks",
        "processing_skips": processing_skips,
        "respiratory_analysis_source": "signals/smoothed",
        "respiratory_analysis_policy": "All future respiratory analyses start from smoothed and respect valid/interpolated masks; no second general smoothing pass",
        "matlab_difference": "All respiratory reference and spectral inputs use smoothed; MATLAB also uses unsmoothed recordings for some calculations",
        "quality_policy": "Only selected QC channels with explicitly inspected support and usable ratings are eligible",
        "automatic_cleaning": {"signature": prepared["signature"], **prepared["provenance"]},
        "polarity": {"assessment": prepared["polarity"], "decision": decision, "applied_to": choice},
        "limitations": [
            "No MATLAB numerical comparison performed",
            "Review coverage may be partial",
            "Weak, unavailable and unassessed channels are ineligible",
            "Automatic candidates are heuristic; interpolation is estimated, not newly inspected evidence",
            "One smoothed respiratory source is an intentional DBAT design; no MATLAB numerical equivalence is claimed",
        ],
        "counts": {
            name: {key: int(mask[:, column].sum()) for key, mask in masks.items()}
            for column, name in enumerate(recording.channels)
        },
    }
    result = PreprocessingResult(recording, preprocessed, smoothed, masks, qc, provenance)
    result.resampled, provenance["resampling"] = resample_views(result, settings["resampling"])
    logger.debug("Smoothing: %s | Detrend: %s", provenance["smoothing"], provenance["detrend"])
    logger.debug("Resampling: %s", provenance["resampling"])
    if qc["review_status"] == "not_reviewed":
        logger.debug("No inspected QC evidence: numerical values are retained, selected filtering is deferred and valid samples remain zero.")
    return result
