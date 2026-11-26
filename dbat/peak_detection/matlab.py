"""Load retained MATLAB respiratory landmarks into the DBAT landmark schema.

Cache fields and one-based index conventions come from the MIT-licensed source:
https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/load_or_extract_respiratory_cycles.m
https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/recompute_respiration_breath_fields.m
"""

import copy
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..config import BELTS
from ..io.recording import source_sha256
from ..logging import logger
from ..preprocessing.annotations import recording_identity
from .respiratory import DetectorSettings, _indices, detect_belt


def load_matlab_peaks(path, preprocessed=None):
    """Read a standard MATLAB *_features.mat file without rerunning detection.

    Parameters
    ----------
    path : path-like
        Cache containing resp_cycles and feature_cache_meta, including x0.
    preprocessed : PreprocessedRecording, optional
        DBAT reference for identity/native-grid checks and QC support. Its final
        waveform is authoritative for imported landmark values and subsequent edits.

    Returns
    -------
    dict
        Same core structure as detect_preprocessed(): identity, fs_hz, time_sec,
        source, preprocessing, review_status, downstream_stale and per-belt data.
        Peaks/troughs and overrides become zero-based; saved troughs are retained.
        Missing raw-source hashes remain unknown. Standalone loads have only
        finite-data support; they do not establish preprocessing QC. With a DBAT
        reference, invalid peaks are removed and unsupported troughs stay missing.
        Imported landmarks always require DBAT review.
    """
    path = Path(path).resolve(strict=True)
    saved = loadmat(path, simplify_cells=True, variable_names=["resp_cycles", "feature_cache_meta"])
    if not {"resp_cycles", "feature_cache_meta"} <= saved.keys():
        raise ValueError("Expected a MATLAB *_features.mat cache with resp_cycles and feature_cache_meta")
    meta, cycles = saved["feature_cache_meta"], saved["resp_cycles"]
    fs, count = float(meta["fs"]), int(meta["n_samples"])
    if not np.isfinite(fs) or fs <= 0 or count < 2 or count != meta["n_samples"]:
        raise ValueError("MATLAB sampling frequency/sample count is invalid")
    channels = [str(name).replace("\u2082", "2") for name in np.atleast_1d(meta["data_columns"])]
    identity = {
        "recording_id": f"Sub{int(meta['subject'])}_M{int(meta['measurement'])}",
        "subject": int(meta["subject"]),
        "measurement": int(meta["measurement"]),
        "source_sha256": None,
        "n_samples": count,
        "fs_hz": fs,
        "channels": channels,
    }
    if preprocessed is not None:
        preprocessed.require_native_grid()
        reference_identity = recording_identity(preprocessed.recording)
        for key in ("subject", "measurement", "fs_hz", "n_samples", "channels"):
            if identity[key] != reference_identity[key]:
                raise ValueError(f"MATLAB {key} differs from the DBAT recording: {path}")
        identity = reference_identity
    import_info = {
        "path": str(path),
        "sha256": source_sha256(path),
        "cache_version": int(meta["cache_version"]),
        "raw_source_hash_available": False,
        "cache_review_status": cycles.get("provenance", {}).get("review_status", "unknown"),
        "settings_policy": "DBAT defaults for subsequent edits; no detection rerun; cached amplitude method retained",
        "waveform_matches": {},
        "polarity_flipped": {},
    }
    result = {
        "algorithm_version": "dbat_respiratory_landmarks_v2",
        "identity": copy.deepcopy(identity),
        "fs_hz": fs,
        "time_sec": np.arange(count) / fs,
        "source": "matlab_cache",
        "preprocessing": {"matlab_import": import_info},
        "review_status": "imported_needs_review",
        "downstream_stale": True,
        "belts": {},
    }
    settings = DetectorSettings(amp_method=str(meta["amp_method"]))
    for channel, key in zip(BELTS, ("lungs", "diaph"), strict=True):
        cached = cycles[key]
        waveform = np.asarray(cached["x0"], dtype=float).reshape(-1)
        if len(waveform) != count:
            raise ValueError(f"{channel}: MATLAB waveform has a different sample count")
        peaks = _indices(np.asarray(cached["peak_idx"], dtype=float) - 1, count)
        troughs = _indices(np.asarray(cached["trough_idx"], dtype=float) - 1, count, missing=True)
        for name, indices in (("peak_t", peaks), ("trough_t", troughs)):
            times = np.asarray(cached[name], dtype=float).reshape(-1)
            if times.shape != indices.shape or not np.allclose(
                times, indices / fs, rtol=0, atol=1e-9, equal_nan=True
            ):
                raise ValueError(f"{channel}: MATLAB {name} disagrees with indices/fs")
        if not np.isfinite(waveform[peaks]).all():
            raise ValueError(f"{channel}: MATLAB peaks lie on missing samples")
        for name, indices in (("peak_val", peaks), ("trough_val", troughs)):
            finite = np.isfinite(indices)
            expected = np.full(indices.size, np.nan)
            expected[finite] = waveform[indices[finite].astype(int)]
            actual = np.asarray(cached[name], dtype=float).reshape(-1)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=1e-7, atol=1e-9, equal_nan=True
            ):
                raise ValueError(f"{channel}: MATLAB {name} disagrees with its saved waveform")
        candidate = _indices(np.asarray(cached.get("auto_peak_idx", []), dtype=float) - 1, count)
        prom = np.asarray(cached.get("auto_peak_prom", []), dtype=float).reshape(-1)
        qc = cached.get("peak_qc", {})
        removed = _indices(np.asarray(qc.get("removed_peak_idx", []), dtype=float) - 1, count)
        reasons = [
            str(reason) for reason in np.asarray(qc.get("removed_reason", []), dtype=object).reshape(-1)
        ]
        if (
            prom.shape != candidate.shape
            or len(reasons) != len(removed)
            or not np.isin(removed, candidate).all()
        ):
            raise ValueError(f"{channel}: inconsistent MATLAB automatic-peak evidence")
        overrides = np.asarray(cached["trough_overrides"], dtype=float).reshape(-1, 3)
        overrides = _indices(overrides.ravel() - 1, count).reshape(-1, 3)
        pairs = set(zip(peaks[:-1], peaks[1:], strict=True))
        if any((left, right) not in pairs or not left < trough < right for left, right, trough in overrides):
            raise ValueError(f"{channel}: invalid MATLAB trough override anchors")
        support = {}
        current = waveform
        if preprocessed is not None:
            column = preprocessed.recording.channels.index(channel)
            support = {
                name: preprocessed.masks[name][:, column] for name in ("valid", "artifact", "interpolated")
            }
            support["channel_status"] = preprocessed.qc["channel_ratings"][channel]
            current = preprocessed.preproc_signal[:, column]
            matches = bool(
                np.allclose(
                    waveform, preprocessed.preproc_signal[:, column], rtol=1e-7, atol=1e-9, equal_nan=True
                )
            )
            import_info["waveform_matches"][channel] = matches
            if not matches and support["channel_status"] != "unavailable":
                logger.warning(
                    "%s: MATLAB waveform differs from DBAT; imported positions need review on the DBAT waveform",
                    channel,
                )
        # Cache v9 omits the thoracic flip flag; MATLAB treats it as false.
        relative = (
            meta.get("thoracic_polarity_flipped", False)
            if key == "lungs"
            else meta["diaphragm_polarity_flipped"]
        )
        import_info["polarity_flipped"][channel] = bool(relative) ^ bool(meta["absolute_polarity_flipped"])
        if (
            preprocessed is not None
            and support["channel_status"] != "unavailable"
            and import_info["polarity_flipped"][channel]
            != (preprocessed.provenance["polarity"]["applied_to"] == channel)
        ):
            raise ValueError(
                f"{channel}: MATLAB and DBAT polarity differ; select peak_detection.source=python or match the cached orientation"
            )
        belt = detect_belt(
            current,
            fs,
            settings,
            source=f"preproc_signal[{column}]" if preprocessed is not None else f"matlab_cache[{key}]",
            **support,
            landmarks={
                "peak_idx": peaks,
                "trough_idx": troughs,
                "trough_overrides": overrides,
                "automatic": {
                    "candidate_idx": candidate,
                    "candidate_prom": prom,
                    "retained_idx": candidate[~np.isin(candidate, removed)],
                    "removed_idx": removed,
                    "removed_reason": reasons,
                    "engine": "MATLAB cache",
                },
            },
        )
        belt["original_imported"] = copy.deepcopy(
            {name: value for name, value in cached.items() if name != "x0"}
        )
        belt["review_status"] = "imported_needs_review"
        result["belts"][channel] = belt
    if preprocessed is not None:
        result["source"] = "preproc_signal"
        result["preprocessing"].update(
            copy.deepcopy(
                {
                    key: preprocessed.provenance[key]
                    for key in ("algorithm_version", "smoothing", "filtering", "resampling", "polarity")
                    if key in preprocessed.provenance
                }
            )
        )
        result["preprocessing"]["qc"] = copy.deepcopy(preprocessed.qc)
    logger.info("Loaded MATLAB respiratory landmarks: %s", path)
    return result
