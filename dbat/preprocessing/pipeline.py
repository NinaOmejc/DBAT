"""Build primary processed signals with independent numerical and QC support."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d

from ..config import BELTS, window_samples
from ..io.recording import Recording
from ..logging import logger
from .annotations import annotation_masks, continuous_runs, new_qc_annotations, recording_identity
from .cleaning import clean_recording, preprocess_spo2 as preprocess_spo2
from .filtering import filter_belt
from .polarity import apply_polarity, assess_relative_belt_polarity, saved_polarity_decision
from .resampling import resample_preprocessed
from .smoothing import moving_mean, smooth_belt


@dataclass
class PreprocessingResult:
    """Immutable native original and one primary preprocessed/smoothed sample grid."""

    recording: Recording
    preprocessed: np.ndarray
    smoothed: np.ndarray
    processed_fs_hz: float
    processed_time_sec: np.ndarray
    masks: dict
    qc: dict
    provenance: dict

    @property
    def original(self):
        """Return immutable supplied native samples without making a copy."""
        return self.recording.samples

    @property
    def original_fs_hz(self):
        """Return the original sampling rate in Hz."""
        return self.recording.fs_hz

    @property
    def original_time_sec(self):
        """Return the original zero-origin native time axis in seconds."""
        return self.recording.time_sec

    @property
    def respiratory_signal(self):
        """Return smoothed; detectors must use processed_fs_hz and masks['valid']."""
        return self.smoothed

    def require_native_grid(self):
        """Reject direct historical native landmark-index reuse on a changed grid.

        Call before transferring native MATLAB indices; identity, indexing-base,
        polarity and waveform checks are also required by a future importer.
        No index conversion or landmark import is performed here.
        """
        if self.processed_fs_hz != self.original_fs_hz or not np.array_equal(
            self.processed_time_sec, self.original_time_sec
        ):
            raise ValueError(
                "Native landmark indices cannot be reused on a different processed grid"
            )


def prepare_recording(recording, config):
    """Run automatic cleaning and source-oriented polarity assessment once.

    Returns corrected native data, automatic masks/events, a preparation
    signature and diagnostics. No manual evidence or polarity flip is inferred.
    """
    logger.info(
        "  Automatic correction and optional polarity assessment for %s...", recording.recording_id
    )
    prepared = clean_recording(recording, config)
    reliable = (
        np.isfinite(prepared["corrected"])
        & ~prepared["masks"]["interpolated"]
        & ~prepared["masks"]["auto_candidate"]
    )
    prepared["polarity"] = assess_relative_belt_polarity(
        prepared["corrected"],
        recording.channels,
        recording.fs_hz,
        reliable,
        config["preprocessing"]["check_relative_belt_polarity"],
    )
    logger.debug("Automatic cleaning: %s", prepared["provenance"]["channels"])
    logger.debug("Relative polarity: %s", prepared["polarity"])
    return prepared


def preprocess_belt(values, artifact, interpolated, fs_hz, detrend, filtering):
    """Detrend/filter a native belt independently of ratings and reviewed coverage.

    Parameters
    ----------
    values : ndarray, shape (native_samples,)
        Corrected, polarity-oriented input, never modified.
    artifact, interpolated : ndarray of bool
        Confirmed native artifacts and automatic interpolation influence.
    fs_hz : float
        Native sampling rate in Hz.
    detrend, filtering : dict
        Detrend parameters and separate Butterworth high/low-pass settings.

    Returns
    -------
    tuple
        Numerical values, support, interpolation influence, filter provenance,
        and skipped detrend runs. Detrending splits at gaps/artifacts; filtering
        follows each finite waveform across QC flags. Ratings do not split either.
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
    output, filter_support, influence, filter_details = filter_belt(
        output, allowed, influence, fs_hz, filtering
    )
    return output, support & filter_support, influence, filter_details, skips


def preprocess_ecg(values):
    """Pass through ECG; TODO: validated ECG-specific conditioning, beyond resampling."""
    return values.copy()


def preprocess_bp(values):
    """Pass through BP; TODO: validated BP-specific conditioning, beyond resampling."""
    return values.copy()


def preprocess_recording(recording, config, qc_annotations=None, prepared=None):
    """Correct, detrend/filter natively, optionally resample once, then smooth once.

    Parameters
    ----------
    recording : Recording
        Immutable native samples and declared native frequency in Hz.
    config : dict
        Validated grouped settings. Optional filters and resampling default off.
    qc_annotations : dict, optional
        Native manual QC; absent review never becomes approved analysis evidence.
    prepared : dict, optional
        Cached automatic correction/polarity result shared with initial QC.

    Returns
    -------
    PreprocessingResult
        Original on the native grid; preprocessed/smoothed and every mask on one
        explicit processed grid. Numerical conditioning ignores ratings/reviewed
        coverage. Future respiratory detection uses smoothed and processed_fs_hz.
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
    if (
        prepared["identity"] != recording_identity(recording)
        or prepared["corrected"].shape != recording.samples.shape
    ):
        raise ValueError("Prepared signals do not match the source identity/native grid")
    decision = saved_polarity_decision(qc, prepared)
    choice = decision["choice"] if decision else None
    if (
        prepared["polarity"]["polarity_warning"]
        and choice is None
        and not (decision and decision["acknowledged_warning"])
    ):
        raise ValueError(
            "Polarity warning requires an explicit reviewer choice or acknowledgement in the QC GUI"
        )
    corrected = apply_polarity(prepared["corrected"], recording.channels, choice)
    masks.update(prepared["masks"])
    masks["finite"] = np.isfinite(corrected)
    usable = np.array([qc["channel_ratings"][name] == "usable" for name in recording.channels])
    unknown = np.array(
        [qc["channel_ratings"][name] in {"weak", "unassessed"} for name in recording.channels]
    )
    selected = np.array([name in config["qc"]["channels"] for name in recording.channels])
    masks["quality_selected"] = np.broadcast_to(selected, corrected.shape).copy()
    masks["uncertain"] = ~masks["reviewed"] | unknown[None, :] | ~masks["quality_selected"]
    masks["eligible"] = (
        masks["quality_selected"]
        & masks["finite"]
        & masks["reviewed"]
        & ~masks["artifact"]
        & usable[None, :]
    )
    settings = config["preprocessing"]
    detrend_window = window_samples(settings["detrend"]["window_sec"], "seconds", recording.fs_hz)
    detrend_window += detrend_window % 2 == 0
    detrend = settings["detrend"]["method"] if settings["enabled"] else "none"
    detrend_parameters = {
        "method": detrend,
        "fs_hz": recording.fs_hz,
        "window_samples": detrend_window if detrend != "none" else None,
        "mean_offset": "per_finite_non_artifact_run" if detrend != "none" else None,
    }
    filtering = {
        name: {**options, "enabled": options["enabled"] and settings["enabled"]}
        for name, options in settings["filtering"].items()
    }
    preprocessed = corrected.copy()
    masks["preprocessed_support"] = masks["finite"] & ~masks["artifact"]
    masks["preprocessed_interpolated"] = masks["interpolated"].copy()
    processors, filter_details, processing_skips = {}, {}, {}
    for column, name in enumerate(recording.channels):
        if name in BELTS and name in settings["channels"]:
            (
                preprocessed[:, column],
                masks["preprocessed_support"][:, column],
                masks["preprocessed_interpolated"][:, column],
                filter_details[name],
                skips,
            ) = preprocess_belt(
                corrected[:, column],
                masks["artifact"][:, column],
                masks["interpolated"][:, column],
                recording.fs_hz,
                detrend_parameters,
                filtering,
            )
            processors[name] = "belt_conditioning"
            processing_skips[name] = {"detrend_native_grid": skips}
        elif name in {"ECG1", "ECG2"}:
            preprocessed[:, column] = preprocess_ecg(corrected[:, column])
            processors[name] = "ecg_todo_passthrough"
        elif name == "Blood Pressure":
            preprocessed[:, column] = preprocess_bp(corrected[:, column])
            processors[name] = "bp_todo_passthrough"
        else:
            processors[name] = (
                "spo2_corrected_without_belt_filters"
                if name == "SpO2"
                else "unselected_passthrough"
            )
    masks["preprocessed_valid"] = masks["eligible"] & masks["preprocessed_support"]
    preprocessed, masks, processed_fs, times, resampling = resample_preprocessed(
        preprocessed, masks, recording.fs_hz, settings["resampling"]
    )
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
    provenance = {
        "algorithm_version": "dbat_preprocessing_v6",
        "enabled": settings["enabled"],
        "selected_channels": settings["channels"],
        "qc_channels": config["qc"]["channels"],
        "channel_processors": processors,
        "detrend": detrend_parameters,
        "filtering": filter_details,
        "smoothing": smoothing_parameters,
        "resampling": resampling,
        "processing_order": "automatic correction; accepted polarity; native detrend; native highpass; native lowpass; optional resampling; one processed-grid smoothing pass",
        "edge_policy": "Moving mean shrinks at recording ends with valid numerical support; genuine gap/artifact influence stays unsupported. Optional IIR filters cross QC/artifact flags; reflection guards at recording ends or genuine missing gaps remain unsupported.",
        "signal_policy": "Numerical conditioning is independent of channel ratings, QC channel selection and reviewed coverage",
        "processing_skips": processing_skips,
        "respiratory_analysis_source": "signals/smoothed",
        "respiratory_analysis_fs_hz": processed_fs,
        "respiratory_analysis_policy": "Every future respiratory detector/analysis uses smoothed, processed_fs_hz and valid/interpolated masks; no second general smoothing pass",
        "native_landmark_indices_compatible": processed_fs == recording.fs_hz
        and np.array_equal(times, recording.time_sec),
        "native_landmark_policy": "Require exact native-grid equality before direct historical MATLAB index reuse; otherwise reject direct reuse",
        "matlab_difference": "All respiratory reference and spectral inputs use smoothed; MATLAB also uses unsmoothed recordings for some calculations",
        "quality_policy": "Only selected QC channels with explicitly inspected support and usable ratings are eligible",
        "automatic_cleaning": {"signature": prepared["signature"], **prepared["provenance"]},
        "polarity": {
            "assessment": prepared["polarity"],
            "decision": decision,
            "applied_to": choice,
        },
        "limitations": [
            "No MATLAB numerical comparison performed",
            "Review coverage may be partial",
            "Automatic candidates are heuristic; interpolation is estimated, not newly inspected evidence",
        ],
        "counts": {
            name: {key: int(mask[:, column].sum()) for key, mask in masks.items()}
            for column, name in enumerate(recording.channels)
        },
    }
    logger.debug("Smoothing: %s | Detrend: %s", smoothing_parameters, detrend_parameters)
    logger.debug("Resampling: %s | Filtering: %s", resampling, filter_details)
    return PreprocessingResult(
        recording, preprocessed, smoothed, processed_fs, times, masks, qc, provenance
    )
