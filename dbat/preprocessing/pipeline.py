"""Orchestrate correction, polarity, native conditioning, resampling and smoothing."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from ..io.recording import (
    Recording,
    load_latest_qc_annotations,
    preprocessing_provenance,
    save_preprocessing_result,
    save_qc_annotations,
)
from ..logging import logger
from .annotations import (
    add_analysis_masks,
    annotation_masks,
    new_qc_annotations,
    recording_identity,
    review_status,
    review_candidates,
    unresolved_candidate_mask,
)
from .cleaning import prepare_recording
from .filtering import detrend_and_filter_recording
from .polarity import apply_polarity, orient_prepared
from .resampling import resample_preprocessed
from .smoothing import smooth_recording
from .signal_quality import assess_signal_quality
from .impact import preprocessing_impact


@dataclass
class PreprocessedRecording:
    """Immutable native original and one primary preprocessed/smoothed sample grid."""

    recording: Recording
    preprocessed: np.ndarray
    preproc_signal: np.ndarray
    processed_fs_hz: float
    processed_time_sec: np.ndarray
    masks: dict
    qc: dict
    provenance: dict
    signal_quality: dict

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


def preprocess_recording(
    recording, config, qc_annotations=None, prepared=None, *, provisional=False
):
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
        Cached automatic correction/polarity result shared with QC.
    provisional : bool, optional
        Preview private, unaccepted edits and unresolved polarity without saving.
        Final processing still requires accepted evidence and polarity resolution.

    Returns
    -------
    PreprocessedRecording
        Original on the native grid; preprocessed/smoothed and every mask on one
        explicit processed grid. Numerical conditioning ignores ratings/reviewed
        coverage. Future respiratory detection uses preproc_signal and processed_fs_hz.
    """
    if (
        recording.fs_hz != config["input"]["fs_hz"]
        or list(recording.channels) != config["input"]["channels"]
    ):
        raise ValueError("Recording fs/channel mapping disagrees with configuration")
    logger.info("  Preprocessing %s...", recording.recording_id)
    qc = qc_annotations if qc_annotations is not None else new_qc_annotations(recording)
    if provisional:
        qc = deepcopy(qc)
        qc["review_status"] = review_status(qc)
    masks = annotation_masks(qc, recording, allow_unaccepted=provisional)
    prepared = prepare_recording(recording, config) if prepared is None else prepared
    if (
        prepared["identity"] != recording_identity(recording)
        or prepared["corrected"].shape != recording.samples.shape
    ):
        raise ValueError(
            "Prepared signals do not match the source identity/native grid"
        )
    corrected, decision = orient_prepared(
        prepared, qc, recording.channels, provisional=provisional
    )
    add_analysis_masks(
        masks,
        corrected,
        recording,
        qc,
        config["qc"]["channels"],
        {
            **prepared["masks"],
            "unresolved_candidate": unresolved_candidate_mask(qc, prepared),
        },
    )
    settings = config["preprocessing"]

    preprocessed, diagnostics, detrended = detrend_and_filter_recording(
        corrected, recording, settings, masks
    )
    preprocessed, masks, processed_fs, times, resampling = resample_preprocessed(
        preprocessed, masks, recording.fs_hz, settings["resampling"]
    )
    diagnostics["resampling"] = resampling
    smoothed = smooth_recording(
        preprocessed, recording, processed_fs, settings, masks, diagnostics
    )

    numerical_signature = hashlib.sha256(
        json.dumps([prepared["signature"], settings], sort_keys=True).encode()
    ).hexdigest()
    quality = assess_signal_quality(
        recording,
        corrected,
        smoothed,
        processed_fs,
        settings["signal_quality"],
        numerical_signature,
    )
    candidates = review_candidates(qc, prepared, quality)
    noise_mask = np.zeros(recording.samples.shape, dtype=bool)
    for candidate in quality["candidates"]:
        column = recording.channels.index(candidate["channel"])
        noise_mask[candidate["start_sample"] : candidate["end_sample"], column] = True
    native_index = np.minimum(
        np.floor(times * recording.fs_hz + 1e-9).astype(int), len(recording.samples) - 1
    )
    masks["noise_candidate"] = noise_mask[native_index]
    masks["unresolved_noise_candidate"] = unresolved_candidate_mask(
        qc, prepared, candidates, source_stage="signal_quality"
    )[native_index]
    result = PreprocessedRecording(
        recording,
        preprocessed,
        smoothed,
        processed_fs,
        times,
        masks,
        qc,
        diagnostics,
        quality,
    )
    result.provenance = preprocessing_provenance(result, config, prepared, decision)
    result.provenance["provisional"] = provisional
    oriented_original = apply_polarity(
        recording.samples, recording.channels, decision["choice"] if decision else None
    )
    result.provenance["preprocessing_impact"] = preprocessing_impact(
        recording,
        oriented_original,
        corrected,
        detrended,
        smoothed,
        processed_fs,
    )
    result.provenance["preprocessing_impact"]["processing_settings"] = {
        "detrend": diagnostics["detrend"],
        "filtering": settings["filtering"],
        "smoothing": diagnostics["smoothing"],
        "resampling": diagnostics["resampling"],
        "artifacts": settings["artifacts"],
        "applied_polarity": decision["choice"] if decision else None,
    }
    return result


def run_preprocessing(recording, config):
    """Load accepted QC, optionally review, then save a new immutable run.

    Returns (PreprocessedRecording, HDF5 path). Every invocation computes the
    current pipeline; no preprocessing cache or old-format reuse is performed.
    Accepted QC is preserved before final recomputation. Cancellation returns
    (None, None); a failure leaves its pending QC recoverable.
    """
    qc = load_latest_qc_annotations(recording, config)
    qc.pop("loaded_from_file", None)
    review_again = config["qc"]["open_gui"] and config["qc"]["review_again"]
    prepared = prepare_recording(recording, config)
    decision = qc.get("polarity")
    needs_review = (
        not qc["accepted_at_utc"]
        or review_again
        or (decision and decision["preparation_signature"] != prepared["signature"])
        or (prepared["polarity"]["polarity_warning"] and not decision)
    )
    if config["qc"]["open_gui"] and needs_review:
        from .preproc_gui import run_qc_gui

        preview = preprocess_recording(recording, config, qc, prepared=prepared, provisional=True)
        qc = run_qc_gui(recording, config, qc, prepared=prepared, preview=preview)
        if qc is None:
            logger.info("Cancelled %s; no result saved", recording.recording_id)
            return None, None
    pending = None
    if qc["accepted_at_utc"]:
        pending = save_qc_annotations(qc, recording, config["paths"]["output_dir"]).parent
    result = preprocess_recording(recording, config, qc, prepared=prepared)
    path = save_preprocessing_result(result, config, run_dir=pending)
    return result, path
