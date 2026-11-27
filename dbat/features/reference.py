"""Stage 5a: static references on the saved DBAT processed grid.

Adapted from Nina Omejc's MIT-licensed MATLAB methodology (2026):
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/utils/get_session_reference_interval.m
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/feature_extraction/compute_respiratory_reference.m
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/feature_extraction/compute_spo2_reference.m
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/utils/robust_resp_excursion.m
Copyright (c) 2026 Nina Omejc. See the repository's MIT LICENSE.
Only static calculations are ported; DBAT QC and preproc_signal are authoritative.
"""

import copy
import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path

import numpy as np

from ..config import BELTS, validate_reference_config
from ..io.recording import load_preprocessing_result, source_sha256
from ..peak_detection.io import load_landmarks
from ..peak_detection.respiratory import validate_belt

SCHEMA = "dbat_static_physiological_reference_v1"


def get_session_reference_interval(n_samples, fs_hz, measurement, reference):
    """Resolve MATLAB's protocol interval using zero-based, half-open bounds.

    Parameters
    ----------
    n_samples : int
        Full recording length on the processed grid.
    fs_hz : float
        Processed sampling frequency in Hz; origin is zero seconds.
    measurement : int
        M1/M3 select pre-stress; M2/M4 select post-stress.
    reference : dict
        Validated [reference] configuration, with minute bounds.

    Returns
    -------
    dict
        Requested/actual indices, times (seconds), duration and interval quality.
        Only M2 ending before the start uses its final min(180 s, duration).
    """
    if (isinstance(n_samples, bool) or not isinstance(n_samples, Real)
            or not math.isfinite(n_samples) or n_samples < 0 or n_samples != int(n_samples)):
        raise ValueError("n_samples must be a nonnegative integer")
    n_samples = int(n_samples)
    if (isinstance(fs_hz, bool) or not isinstance(fs_hz, Real)
            or not math.isfinite(fs_hz) or fs_hz <= 0):
        raise ValueError("fs_hz must be finite and positive")
    if (isinstance(measurement, bool) or not isinstance(measurement, Real)
            or measurement not in (1, 2, 3, 4)):
        raise ValueError("Reference intervals are defined only for M1, M2, M3 and M4")
    validate_reference_config(reference)
    phase = "pre" if measurement in (1, 3) else "post"
    # MATLAB round for nonnegative sample positions: ties round away from zero.
    start, stop = (math.floor(60 * reference[f"{phase}_{edge}_min"] * fs_hz + 0.5)
                   for edge in ("start", "end"))
    interval = {
        "reference_start_idx": 0, "reference_end_idx": 0,
        "reference_start_t": None, "reference_end_t": None, "reference_duration_sec": 0.0,
        "requested_start_idx": start, "requested_end_idx": stop,
        "requested_start_t": start / fs_hz, "requested_end_t": stop / fs_hz,
        "requested_duration_sec": (stop - start) / fs_hz,
        "protocol_phase": f"{phase}_stress", "measurement": int(measurement),
        "available": False, "complete": False, "truncated": False,
        "quality": "unavailable", "truncation_reason": "",
    }
    fallback = measurement == 2 and 0 < n_samples <= start
    if n_samples <= start and not fallback:
        interval.update(truncated=True, quality="recording_ends_before_reference_start",
                        truncation_reason="recording_ends_before_reference_start")
        return interval
    first, end = (max(0, n_samples - max(1, math.floor(180 * fs_hz + 0.5))), n_samples) \
        if fallback else (start, min(n_samples, stop))
    complete = not fallback and end == stop
    interval.update(
        reference_start_idx=int(first), reference_end_idx=int(end),
        reference_start_t=first / fs_hz, reference_end_t=end / fs_hz,
        reference_duration_sec=(end - first) / fs_hz,
        available=end > first, complete=complete, truncated=not complete,
        quality="fallback_last_3_minutes" if fallback else ("good" if complete else "truncated_recording"),
        truncation_reason="recording_ends_before_reference_start" if fallback else (
            "" if complete else "recording_ends_inside_reference_interval"),
    )
    return interval


def _summary(values, valid, minimum, value_name, quality, *, breaths=False):
    """Summarize supported evidence; unavailable numeric values stay JSON null."""
    count, total = int(np.count_nonzero(valid)), len(values)
    available = not quality and count >= minimum
    result = {
        "available": available,
        "quality": quality or ("good" if available else (
            "insufficient_breaths" if breaths else "insufficient_valid_samples")),
        "n_valid_breaths" if breaths else "n_valid_samples": count,
        "n_input_breaths" if breaths else "n_interval_samples": total,
        "valid_fraction": count / total if total else 0.0,
        "minimum_valid_breaths" if breaths else "minimum_valid_samples": minimum,
        value_name: float(np.median(values[valid])) if available and value_name != "excursion" else None,
    }
    return result


def calculate_references(data, landmarks, reference):
    """Calculate once for a full recording, without detecting or smoothing again.

    Parameters
    ----------
    data : dict
        load_preprocessing_result output, including final waveform and QC masks.
    landmarks : dict
        Matching manually reviewed DBAT landmarks with saved amp/breath_valid.
    reference : dict
        [reference] settings; no feature-window bounds are accepted.

    Returns
    -------
    dict
        Common interval, independent belt session/global amplitudes and waveform
        excursions, and SpO2 median/mean. Missing values are None (JSON null).
        Coverage denominators include all actual interval samples/input breaths.
    """
    values, time = np.asarray(data["preproc_signal"]), np.asarray(data["processed_time_sec"])
    fs, channels, masks = data["processed_fs_hz"], data["channels"], data["masks"]
    metadata = data["metadata"]
    if (data["respiratory_analysis_source"] != "preproc_signal"
            or landmarks["source"] != "preproc_signal"
            or landmarks["identity"] != metadata["identity"]
            or landmarks["fs_hz"] != fs
            or values.shape != (len(time), len(channels))
            or not np.array_equal(landmarks["time_sec"], time)
            or not np.isfinite(fs) or fs <= 0
            or not np.allclose(time, np.arange(len(time)) / fs, rtol=0, atol=1e-9)
            or landmarks["preprocessing"]["qc"] != metadata["qc"]):
        raise ValueError("Landmarks and preprocessing identity, waveform source, QC or grid disagree")
    if landmarks["review_status"] != "manual_reviewed":
        raise ValueError("References require successful manual landmark review")
    for name in ("valid", "artifact", "uncertain"):
        if np.asarray(masks[name]).dtype != bool or masks[name].shape != values.shape:
            raise ValueError(f"Expected a Boolean processed-grid {name} mask")
    interval = get_session_reference_interval(len(time), fs, metadata["identity"]["measurement"], reference)
    section = slice(interval["reference_start_idx"], interval["reference_end_idx"])
    result = {
        "schema_version": SCHEMA, "identity": copy.deepcopy(metadata["identity"]),
        "processed_fs_hz": float(fs), "n_processed_samples": len(time),
        "index_convention": "zero_based_half_open", "signal_source": "preproc_signal",
        "settings": copy.deepcopy(reference), "interval": interval, "respiratory": {},
        "warning_action": "retain_data_no_correction",
    }
    for channel in (*BELTS, "SpO2"):
        status = metadata["qc"]["channel_ratings"].get(channel, "unassessed")
        quality = "channel_missing" if channel not in channels else (
            "channel_unavailable" if status == "unavailable" else (
                "channel_not_usable" if status != "usable" else ""))
        if channel in channels:
            column = channels.index(channel)
            x = values[:, column]
            support = masks["valid"][:, column] & ~masks["artifact"][:, column] \
                & ~masks["uncertain"][:, column] & np.isfinite(x)
        else:
            x, support = np.full(len(time), np.nan), np.zeros(len(time), bool)
        if quality:
            support = np.zeros(len(time), bool)
        interval_quality = quality or ("" if interval["available"] else "reference_interval_unavailable")
        segment, valid = x[section], support[section]
        if channel == "SpO2":
            spo2 = _summary(segment, valid, reference["spo2_min_valid_samples"],
                            "median_percent", interval_quality)
            spo2["mean_percent"] = float(np.mean(segment[valid])) if spo2["available"] else None
            result["spo2"] = spo2
            continue
        raw = _summary(segment, valid, 3, "excursion", interval_quality)
        raw["minimum_valid_fraction"] = reference["resp"]["raw_min_finite_fraction"]
        if raw["available"]:
            if raw["valid_fraction"] < raw["minimum_valid_fraction"]:
                raw.update(available=False, quality="insufficient_valid_coverage", excursion=None)
            else:
                low, high = np.percentile(segment[valid], [5, 95], method="hazen")
                excursion = float(high - low)
                if not np.isfinite(excursion) or excursion <= 0:
                    raw.update(available=False, quality="invalid_excursion", excursion=None)
                else:
                    raw["excursion"] = excursion
        belt = landmarks["belts"].get(channel)
        breath_quality = quality or ("landmarks_unavailable" if belt is None else "")
        if belt is None:
            peak_t, amplitude, breath_valid = np.array([]), np.array([]), np.array([], dtype=bool)
        else:
            validate_belt(belt, fs)  # Validates dependency fingerprint, never recomputes fields.
            if (belt["review_status"] != "manual_reviewed" or channel not in channels
                    or belt["channel_status"] != status
                    or not np.array_equal(belt["source_waveform"], x, equal_nan=True)
                    or not np.array_equal(belt["detection_waveform"], x, equal_nan=True)
                    or not np.array_equal(belt["sample_valid"], masks["valid"][:, column])
                    or not np.array_equal(belt["artifact_mask"], masks["artifact"][:, column])):
                raise ValueError(f"{channel}: reviewed landmarks do not match preprocessing")
            peak_t, amplitude, breath_valid = (np.asarray(belt[key]) for key in ("peak_t", "amp", "breath_valid"))
            if (peak_t.ndim != 1 or amplitude.shape != peak_t.shape
                    or breath_valid.shape != peak_t.shape or breath_valid.dtype != bool):
                raise ValueError(f"{channel}: amplitudes and breath validity must align with peaks")
            breath_valid = breath_valid & support[belt["peak_idx"]]
        valid_breaths = breath_valid & np.isfinite(peak_t) & np.isfinite(amplitude) & (amplitude > 0)
        if breath_quality:
            valid_breaths = np.zeros(len(amplitude), bool)
        in_interval = (peak_t >= interval["reference_start_t"]) & (peak_t < interval["reference_end_t"]) \
            if interval["available"] else np.zeros(len(peak_t), bool)
        session = _summary(amplitude[in_interval], valid_breaths[in_interval], reference["resp_min_breaths"],
                           "value", breath_quality or interval_quality, breaths=True)
        global_reference = _summary(amplitude, valid_breaths, 1, "value", breath_quality, breaths=True)
        result["respiratory"][channel] = {
            "session": session, "global": global_reference, "waveform": raw,
            "amplitude_method": belt["settings"]["amp_method"] if belt is not None else None,
        }
    summaries = [result["spo2"]]
    for belt in result["respiratory"].values():
        summaries.extend((belt["session"], belt["waveform"]))
    for summary in summaries:
        if summary["available"] and not interval["complete"]:
            summary["quality"] = "warning_truncated_interval"
    return result


def calculate_saved_references(preprocessing_path, landmark_path, reference, output_dir=None):
    """Validate saved revisions, calculate and publish a separate immutable JSON result.

    Parameters
    ----------
    preprocessing_path, landmark_path : path-like
        Matching DBAT HDF5 revisions. Landmark provenance must name the exact
        preprocessing file hash; moved files with identical bytes are accepted.
    reference : dict
        [reference] settings. Raw recordings and GUIs are never opened.
    output_dir : path-like, optional
        Defaults to references/ alongside the landmark output root.

    Returns
    -------
    (dict, pathlib.Path)
        References and new numbered JSON revision, with input hashes/provenance.
        Existing preprocessing, landmarks and reference revisions are unchanged.
    """
    preprocessing_path, landmark_path = (Path(path).resolve(strict=True)
                                         for path in (preprocessing_path, landmark_path))
    preprocessing_hash, landmark_hash = source_sha256(preprocessing_path), source_sha256(landmark_path)
    data, landmarks = load_preprocessing_result(preprocessing_path), load_landmarks(landmark_path)
    if landmarks["preprocessing"].get("result_sha256") != preprocessing_hash:
        raise ValueError("Landmark revision does not reference this exact saved preprocessing revision")
    result = calculate_references(data, landmarks, reference)
    if (source_sha256(preprocessing_path) != preprocessing_hash
            or source_sha256(landmark_path) != landmark_hash):
        raise ValueError("Input revisions changed during reference calculation")
    result["provenance"] = {
        "preprocessing": {"path": str(preprocessing_path), "sha256": preprocessing_hash,
                          "schema_version": data["schema_version"]},
        "landmarks": {"path": str(landmark_path), "sha256": landmark_hash,
                      "schema_version": landmarks["schema_version"], "revision": landmarks["revision"],
                      "review": landmarks["review_provenance"]},
        "numpy_version": np.__version__,
    }
    recording_id = result["identity"]["recording_id"]
    if re.fullmatch(r"[A-Za-z0-9_-]+", recording_id) is None:
        raise ValueError("Unsafe recording ID")
    output_dir = Path(output_dir) if output_dir is not None else landmark_path.parents[2] / "references"
    folder = output_dir.resolve() / recording_id
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{recording_id}_references_r"
    revisions = [int(path.stem[len(prefix):]) for path in folder.glob(f"{prefix}*.json")
                 if path.stem[len(prefix):].isdigit()]
    result.update(revision=max(revisions, default=0) + 1, saved_at_utc=datetime.now(UTC).isoformat())
    destination = folder / f"{prefix}{result['revision']:04d}.json"
    fd, temporary = tempfile.mkstemp(prefix=".references-", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)  # Atomic publication; refuses an existing destination.
    finally:
        os.unlink(temporary)
    return result, destination
