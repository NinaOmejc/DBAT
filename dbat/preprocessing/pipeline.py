"""Orchestrate correction, polarity, native conditioning, resampling and smoothing."""

from dataclasses import dataclass

import numpy as np

from ..config import BELTS as BELTS, window_samples as window_samples
from ..io.recording import (
    Recording,
    load_latest_qc_annotations,
    load_reusable_preprocessing_result,
    preprocessing_provenance,
    save_preprocessing_result,
    save_qc_annotations,
)
from ..logging import logger
from .annotations import (
    add_analysis_masks,
    annotation_masks,
    continuous_runs as continuous_runs,
    new_qc_annotations,
    recording_identity,
)
from .cleaning import (
    clean_recording as clean_recording,
    prepare_recording as prepare_recording,
    preprocess_bp as preprocess_bp,
    preprocess_ecg as preprocess_ecg,
    preprocess_spo2 as preprocess_spo2,
)
from .filtering import (
    detrend_and_filter_recording,
    filter_belt as filter_belt,
    preprocess_belt as preprocess_belt,
)
from .polarity import (
    apply_polarity as apply_polarity,
    assess_relative_belt_polarity as assess_relative_belt_polarity,
    orient_prepared,
    saved_polarity_decision as saved_polarity_decision,
)
from .resampling import resample_preprocessed
from .smoothing import (
    moving_mean as moving_mean,
    smooth_belt as smooth_belt,
    smooth_recording,
)


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
    corrected, decision = orient_prepared(prepared, qc, recording.channels)
    add_analysis_masks(masks, corrected, recording, qc, config["qc"]["channels"], prepared["masks"])
    settings = config["preprocessing"]

    preprocessed, diagnostics = detrend_and_filter_recording(corrected, recording, settings, masks)
    preprocessed, masks, processed_fs, times, resampling = resample_preprocessed(
        preprocessed, masks, recording.fs_hz, settings["resampling"]
    )
    diagnostics["resampling"] = resampling
    smoothed = smooth_recording(preprocessed, recording, processed_fs, settings, masks, diagnostics)

    result = PreprocessingResult(
        recording, preprocessed, smoothed, processed_fs, times, masks, qc, diagnostics
    )
    result.provenance = preprocessing_provenance(result, config, prepared, decision)
    return result


def preprocess_or_load_recording(recording, config):
    """Reuse compatible output, or load QC, review, preprocess and save a new run.

    Returns (PreprocessingResult, HDF5 path, reused). GUI cancellation returns
    (None, None, False), preserving existing results and accepted QC. Cache hits
    skip preparation and GUI review; cache misses honor qc.open_gui as before.
    """
    qc = load_latest_qc_annotations(recording, config)
    cached = load_reusable_preprocessing_result(recording, config, qc)
    if cached is not None:
        saved, path = cached
        result = PreprocessingResult(
            recording, saved["signals"]["preprocessed"], saved["signals"]["smoothed"],
            saved["processed_fs_hz"], saved["processed_time_sec"], saved["masks"],
            saved["metadata"]["qc"], saved["metadata"]["processing"],
        )
        return result, path, True
    prepared = prepare_recording(recording, config)
    if config["qc"]["open_gui"]:
        from .preproc_gui import run_qc_gui

        qc = run_qc_gui(recording, config, qc, prepared=prepared)
        if qc is None:
            logger.info("Cancelled %s; no result saved", recording.recording_id)
            return None, None, False
        save_qc_annotations(qc, recording, config["paths"]["output_dir"])
    result = preprocess_recording(recording, config, qc, prepared=prepared)
    path = save_preprocessing_result(result, config)
    return result, path, False
