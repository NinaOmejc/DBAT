"""Respiratory landmarks on the already-smoothed DBAT processed grid.

Numerical logic adapted from Nina Omejc's MIT-licensed MATLAB implementation:
https://github.com/NinaOmejc/magma-labeling/tree/a7081d2e87131bb05563431778ad10b2865296a7/src/feature_extraction
Python numerical core reused from:
https://github.com/NinaOmejc/magma-ml/blob/f0fbbf435f892b9c1dd972f62f89d783207ad752/src/landmarks/detection.py
Copyright (c) 2026 Nina Omejc. See the repository LICENSE for the MIT terms.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import find_peaks

from ..preprocessing.annotations import continuous_runs, recording_identity


@dataclass(frozen=True)
class DetectorSettings:
    """MATLAB detector defaults; durations are seconds, amplitudes waveform units."""

    min_peak_dist_sec: float = 1.5
    min_peak_prom: float = 0.5
    min_peak_height: float = -1.0
    trough_method: str = "min"
    trough_prct: float = 5.0
    amp_method: str = "symmetric"
    qc_enabled: bool = True
    short_ibi_ratio: float = 0.65
    rhythm_merge_tol: float = 0.35
    min_prom_ratio: float = 0.35
    local_window_breaths: int = 7
    qc_iterations: int = 5

    def __post_init__(self):
        """Reject ambiguous settings before any waveform or annotation is changed."""
        numbers = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(numbers).all():
            raise ValueError("Detector parameters must be finite")
        if self.min_peak_dist_sec <= 0 or self.min_peak_prom < 0:
            raise ValueError("Invalid peak distance or prominence")
        if self.trough_method not in {"min", "prctile"} or not 0 <= self.trough_prct <= 100:
            raise ValueError("Invalid trough method or percentile")
        if self.amp_method not in {"symmetric", "inspiratory", "expiratory"}:
            raise ValueError("Invalid amplitude method")
        if min(self.short_ibi_ratio, self.rhythm_merge_tol, self.min_prom_ratio) < 0:
            raise ValueError("QC thresholds must be nonnegative")
        for value, lower, upper in (
            (self.local_window_breaths, 3, 1000),
            (self.qc_iterations, 0, 5),
        ):
            if not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError("QC needs an integer neighborhood >=3 and 0..5 iterations")


def waveform_hash(values):
    """Hash waveform shape and float64 samples in a platform-independent order."""
    array = np.asarray(values, dtype="<f8")
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _indices(values, count, *, missing=False):
    """Validate zero-based vectors, allowing NaN trough slots when requested."""
    array = np.asarray(values, dtype=float)
    if array.ndim > 2 or (array.ndim == 2 and min(array.shape, default=0) > 1):
        raise ValueError("Landmark indices must be vectors")
    array = array.ravel()
    finite = np.isfinite(array)
    if (
        (not missing and not finite.all())
        or np.isinf(array).any()
        or np.any(array[finite] != np.floor(array[finite]))
        or np.any((array[finite] < 0) | (array[finite] >= count))
    ):
        raise ValueError("Landmarks must be integer input-grid sample indices within the waveform")
    return array if missing else array.astype(np.int64)


def peak_qc(peaks, prominences, fs, settings):
    """Apply MATLAB's simultaneous local-rhythm removals for at most five passes.

    Returns retained peaks/prominences and removed samples/reasons. The local
    IBI window deliberately includes the candidate's adjacent intervals, as MATLAB does.
    Adapted from:
    https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/feature_extraction/extract_respiration_feature.m
    """
    peaks, prom = np.asarray(peaks).copy(), np.asarray(prominences).copy()
    removed, reasons = [], []
    for _ in range(settings.qc_iterations if settings.qc_enabled else 0):
        if peaks.size < 3:
            break
        ibi = np.diff(peaks) / fs
        drop = np.zeros(peaks.size, bool)
        iteration_reasons = {}
        half = settings.local_window_breaths // 2
        for i in range(peaks.size):
            neighbors = prom[max(0, i - half) : min(prom.size, i + half + 1)].copy()
            neighbors[i - max(0, i - half)] = np.nan
            neighbors = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
            intervals = ibi[max(0, i - half) : min(ibi.size, i + half + 1)]
            if not neighbors.size or not intervals.size or not np.isfinite(prom[i]):
                continue
            rhythm = np.median(intervals)
            short = (i > 0 and ibi[i - 1] < settings.short_ibi_ratio * rhythm) or (
                i < ibi.size and ibi[i] < settings.short_ibi_ratio * rhythm
            )
            restores = (
                0 < i < ibi.size
                and abs(ibi[i - 1] + ibi[i] - rhythm) <= settings.rhythm_merge_tol * rhythm
            )
            low = prom[i] < settings.min_prom_ratio * np.median(neighbors)
            if short and (restores or low):
                drop[i] = True
                iteration_reasons[i] = "+".join(
                    ["short_ibi"]
                    + (["rhythm"] if restores else [])
                    + (["low_prominence"] if low else [])
                )
        if not drop.any():
            break
        removed.extend(peaks[drop].tolist())
        reasons.extend(iteration_reasons[i] for i in np.flatnonzero(drop))
        peaks, prom = peaks[~drop], prom[~drop]
    return peaks, prom, np.asarray(removed, dtype=int), reasons


def detect_peaks(x, fs, settings):
    """Use SciPy on finite runs; return pre-QC candidates and retained samples.

    SciPy selects plateau midpoints (MATLAB selects the first sample) and applies
    distance before prominence. Keep input-grid SciPy candidates for audit rather
    than claiming exact MATLAB equality on these cases.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Provide a one-dimensional signal and a positive sampling rate")
    candidates, prominences, retained, removed, reasons = [], [], [], [], []
    distance = max(1, int(np.floor(settings.min_peak_dist_sec * fs + 0.5)))
    for first, stop in continuous_runs(np.isfinite(x)):
        locs, properties = find_peaks(
            x[first:stop],
            distance=distance,
            prominence=settings.min_peak_prom,
            height=settings.min_peak_height,
        )
        locs = locs + first
        prom = properties["prominences"]
        kept, _, dropped, why = peak_qc(locs, prom, fs, settings)
        candidates.extend(locs)
        prominences.extend(prom)
        retained.extend(kept)
        removed.extend(dropped)
        reasons.extend(why)
    return {
        "candidate_idx": np.asarray(candidates, dtype=int),
        "candidate_prom": np.asarray(prominences, dtype=float),
        "retained_idx": np.asarray(retained, dtype=int),
        "removed_idx": np.asarray(removed, dtype=int),
        "removed_reason": reasons,
        "engine": "scipy.signal.find_peaks",
    }


def _support_reason(belt, left, right):
    """Explain input-grid support over an inclusive peak pair without changing landmarks."""
    section = slice(left, right + 1)
    reasons = []
    if not np.isfinite(belt["detection_waveform"][section]).all():
        reasons.append("nonfinite")
    if belt["artifact_mask"][section].any():
        reasons.append("confirmed_artifact")
    if belt["uncertain_mask"][section].any():
        reasons.append("uncertain_support")
    if belt["channel_status"] == "unavailable":
        reasons.append("channel_unavailable")
    return "+".join(reasons)


def _dependency_hash(belt, fs):
    """Identify the exact waveform, settings, QC and landmarks behind breath outputs."""
    digest = hashlib.sha256(
        json.dumps([float(fs), belt["settings"], belt["channel_status"]], sort_keys=True).encode(
            "utf-8"
        )
    )
    for key in ("detection_waveform", "peak_idx", "trough_idx", "artifact_mask", "uncertain_mask"):
        digest.update(waveform_hash(belt[key]).encode("ascii"))
    return digest.hexdigest()


def recompute(belt, fs):
    """Derive breath fields in place; retain exact troughs for unchanged peak pairs.

    Missing troughs occupy NaN slots. Confirmed artifacts, uncertainty and
    nonfinite gaps invalidate analytical support, never existing positions.
    Finite trough locations remain numerical evidence even on unreviewed or
    artifact-flagged support; their analytical masks remain false.
    Adapted from:
    https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/feature_extraction/recompute_respiration_breath_fields.m

    Parameters
    ----------
    belt : dict
        Input-grid waveform, zero-based peaks, anchored troughs and sample QC masks.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        The same belt, updated in place with breath fields and support masks.
    """
    x = belt["detection_waveform"]
    settings = DetectorSettings(**belt["settings"])
    peaks = _indices(belt["peak_idx"], x.size)
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("Peak indices must be strictly increasing")
    n = peaks.size
    troughs = np.full(max(n - 1, 0), np.nan)
    reasons = []
    pairs = set(zip(peaks[:-1], peaks[1:], strict=True))
    for key in ("trough_overrides", "preserved_troughs"):
        belt[key] = np.asarray(
            [row for row in belt[key] if tuple(row[:2]) in pairs], dtype=float
        ).reshape(-1, 3)
    pinned = {tuple(row[:2]): row[2] for row in belt["preserved_troughs"]}
    pinned.update({tuple(row[:2]): row[2] for row in belt["trough_overrides"]})
    for i, (left, right) in enumerate(zip(peaks[:-1], peaks[1:], strict=True)):
        reason = _support_reason(belt, left, right)
        if (left, right) in pinned:
            troughs[i] = pinned[left, right]
        elif right - left > 1 and np.isfinite(x[left : right + 1]).all():
            segment = x[left + 1 : right]
            if settings.trough_method == "min":
                offset = np.argmin(segment)
            else:
                # MATLAB's traditional exact prctile uses Hazen plotting positions.
                target = np.percentile(segment, settings.trough_prct, method="hazen")
                offset = np.argmin(np.abs(segment - target))
            troughs[i] = left + 1 + offset
        if not np.isfinite(troughs[i]):
            reason = "+".join(filter(None, (reason, "missing_trough")))
        reasons.append(reason)
    troughs = _indices(troughs, x.size, missing=True)
    finite = np.isfinite(troughs)
    if np.any((troughs[finite] <= peaks[:-1][finite]) | (troughs[finite] >= peaks[1:][finite])):
        raise ValueError("Each trough must lie strictly between its surrounding peaks")
    trough_values = np.full(troughs.size, np.nan)
    trough_values[finite] = x[troughs[finite].astype(int)]
    for i in np.flatnonzero(finite & ~np.isfinite(trough_values)):
        reasons[i] = "+".join(filter(None, (reasons[i], "nonfinite_trough")))
    interval_valid = np.array([not reason for reason in reasons], dtype=bool)
    peak_values = x[peaks]
    exp, insp, sym = (np.full(n, np.nan) for _ in range(3))
    supported_trough = np.where(interval_valid, trough_values, np.nan)
    if n > 1:
        exp[:-1] = peak_values[:-1] - supported_trough
        insp[1:] = peak_values[1:] - supported_trough
    if n > 2:
        sym[1:-1] = peak_values[1:-1] - (supported_trough[:-1] + supported_trough[1:]) / 2
    ibi_raw = np.diff(peaks) / fs
    ibi = np.where(interval_valid, ibi_raw, np.nan)
    rr = 60 / ibi
    good = ibi[np.isfinite(ibi)]
    rates = rr[np.isfinite(rr)]
    amplitude = {"expiratory": exp, "inspiratory": insp, "symmetric": sym}[settings.amp_method]
    breath_reason = []
    for i in range(n):
        adjacent = reasons[max(0, i - 1) : min(i + 1, len(reasons))]
        adjacent.append(_support_reason(belt, peaks[i], peaks[i]))
        reason = ";".join(dict.fromkeys(filter(None, adjacent)))
        if not np.isfinite(amplitude[i]) and not reason:
            reason = "undefined_boundary_amplitude"
        breath_reason.append(reason)
    belt.update(
        peak_idx=peaks,
        peak_t=peaks / fs,
        peak_val=peak_values,
        trough_idx=troughs,
        trough_t=troughs / fs,
        trough_val=trough_values,
        amp_exp=exp,
        amp_insp=insp,
        amp_sym=sym,
        amp=amplitude,
        ibi_raw=ibi_raw,
        ibi=ibi,
        rr_bpm=rr,
        rr_mean_bpm=float(60 / np.mean(good)) if good.size else np.nan,
        rr_std_bpm=float(np.std(rates, ddof=1))
        if rates.size > 1
        else (0.0 if rates.size == 1 else np.nan),
        ibi_valid=interval_valid,
        ibi_reason=reasons,
        breath_valid=np.array([not reason for reason in breath_reason], dtype=bool),
        breath_reason=breath_reason,
        trough_valid=interval_valid.copy(),
        ok=bool(n >= 3 and interval_valid.any()),
    )
    belt["derived_fingerprint"] = _dependency_hash(belt, fs)
    belt["derived_stale"] = False
    belt["source_waveform"].setflags(write=False)
    belt["detection_waveform"].setflags(write=False)
    validate_belt(belt, fs)
    return belt


def validate_belt(belt, fs):
    """Validate ordered indices, times, override anchors and explicit missing slots."""
    count = len(belt["detection_waveform"])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be finite and positive")
    peaks = _indices(belt["peak_idx"], count)
    troughs = _indices(belt["trough_idx"], count, missing=True)
    if np.any(np.diff(peaks) <= 0) or troughs.size != max(0, peaks.size - 1):
        raise ValueError("Expected ordered peaks and one trough slot per interpeak interval")
    finite = np.isfinite(troughs)
    if np.any((troughs[finite] <= peaks[:-1][finite]) | (troughs[finite] >= peaks[1:][finite])):
        raise ValueError("Troughs must be strictly inside adjacent peak pairs")
    for key, indices in (("peak_t", peaks), ("trough_t", troughs)):
        if not np.allclose(belt[key], indices / fs, rtol=0, atol=1e-8, equal_nan=True):
            raise ValueError(f"{key} disagrees with zero-based indices and fs")
    if len(belt["ibi_valid"]) != troughs.size:
        raise ValueError("Interval support does not align with landmarks")
    if np.any(~finite & belt["ibi_valid"]):
        raise ValueError("A missing trough requires unassessable interval support")
    pairs = set(zip(peaks[:-1], peaks[1:], strict=True))
    for left, right, trough in belt["trough_overrides"]:
        if (
            (left, right) not in pairs
            or not left < trough < right
            or int(trough) != trough
            or not np.isfinite(belt["detection_waveform"][int(trough)])
        ):
            raise ValueError("Invalid manual trough override")
    if "derived_fingerprint" in belt and belt["derived_fingerprint"] != _dependency_hash(belt, fs):
        belt["derived_stale"] = True
        raise ValueError(
            "QC, settings or landmarks changed; explicitly recompute stale breath outputs"
        )


def edit_peak(belt, fs, time_sec, *, remove=None):
    """Add the maximum within ±1 s, or remove an exact existing sample index.

    Mutates only after validating the complete proposed state. Exact duplicates
    and insertions within MATLAB's quarter-minimum-distance guard are rejected.
    """
    proposed = copy.deepcopy(belt)
    peaks = proposed["peak_idx"]
    x = proposed["detection_waveform"]
    if remove is not None:
        if remove not in peaks:
            raise ValueError("No peak exists at the selected sample")
        index, action = int(remove), "remove"
        proposed["peak_idx"] = peaks[peaks != index]
    else:
        if not np.isfinite(time_sec) or not 0 <= time_sec <= (len(x) - 1) / fs:
            raise ValueError("Click lies outside the recording")
        center = int(np.floor(time_sec * fs + 0.5))
        radius = max(1, int(np.floor(fs + 0.5)))
        first, stop = max(0, center - radius), min(len(x), center + radius + 1)
        section = x[first:stop]
        if not np.isfinite(section).any():
            raise ValueError("No finite sample within one second of the click")
        index = first + int(np.argmax(np.where(np.isfinite(section), section, -np.inf)))
        guard = max(1, int(np.floor(proposed["settings"]["min_peak_dist_sec"] * fs / 4 + 0.5)))
        if peaks.size and np.min(np.abs(peaks - index)) <= guard:
            raise ValueError("Peak already exists or is too close to an existing peak")
        proposed["peak_idx"] = np.sort(np.r_[peaks, index])
        action = "add"
    proposed["manual_peak_edits"].append(
        {"action": action, "sample": index, "time_sec": index / fs}
    )
    recompute(proposed, fs)
    belt.clear()
    belt.update(proposed)
    return index


def move_trough(belt, fs, left, right, time_sec=None, *, reset=False):
    """Move/reset one pair's trough, snapping to a input-grid sample without searching."""
    peaks = belt["peak_idx"]
    if not np.any((peaks[:-1] == left) & (peaks[1:] == right)):
        raise ValueError("The surrounding peaks are no longer consecutive")
    proposed = copy.deepcopy(belt)
    for key in ("trough_overrides", "preserved_troughs"):
        rows = proposed[key]
        proposed[key] = rows[~((rows[:, 0] == left) & (rows[:, 1] == right))]
    if not reset:
        if time_sec is None or not np.isfinite(time_sec):
            raise ValueError("Provide a finite trough time in seconds")
        index = int(np.floor(time_sec * fs + 0.5))
        if not left < index < right or not np.isfinite(proposed["detection_waveform"][index]):
            raise ValueError("Trough must be finite and strictly between its two peaks")
        proposed["trough_overrides"] = np.vstack(
            [proposed["trough_overrides"], [left, right, index]]
        )
    proposed["manual_trough_edits"].append(
        {
            "action": "reset" if reset else "move",
            "left": int(left),
            "right": int(right),
            "sample": None if reset else index,
        }
    )
    recompute(proposed, fs)
    belt.clear()
    belt.update(proposed)


def detect_belt(
    values,
    fs_hz,
    settings=None,
    *,
    valid=None,
    artifact=None,
    interpolated=None,
    source="array",
    channel_status="unknown",
):
    """Detect respiratory landmarks without applying another smoothing pass.

    Parameters
    ----------
    values : ndarray, shape (processed_samples,)
        Already-smoothed belt samples on the supplied grid, left unchanged.
    fs_hz : float
        Actual sampling frequency of this waveform in Hz.
    settings : DetectorSettings, optional
        Peak selection, conservative QC and trough/amplitude definitions.
    valid, artifact, interpolated : ndarray of bool, optional
        Independent analysis support, technical artifact and interpolation masks.
        QC changes analytical support, while finite landmarks remain evidence.
    source : str
        Waveform provenance; standalone input defaults to "array".
    channel_status : str
        Recording-level reliability; "unavailable" disables analytical support.

    Returns
    -------
    dict
        Automatic candidates, accepted peaks/troughs, timing, amplitudes,
        support masks and reasons. All indices are zero-based on this input grid.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("Provide a one-dimensional waveform and positive fs_hz")
    settings = settings or DetectorSettings()
    masks = {}
    for name, supplied, default in (
        ("sample_valid", valid, np.isfinite(x)),
        ("artifact_mask", artifact, np.zeros(x.size, bool)),
        ("sample_interpolated", interpolated, np.zeros(x.size, bool)),
    ):
        mask = np.asarray(default if supplied is None else supplied)
        if mask.shape != x.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{name} must be binary on the waveform's sample grid")
        masks[name] = mask.astype(bool).copy()
    automatic = detect_peaks(x, fs_hz, settings)
    for name in ("candidate", "retained", "removed"):
        automatic[f"{name}_t"] = automatic[f"{name}_idx"] / fs_hz
    automatic["candidate_val"] = x[automatic["candidate_idx"]]
    belt = {
        "source_waveform": x.copy(),
        "detection_waveform": x.copy(),
        "source": source,
        "fs_hz": float(fs_hz),
        "source_hash": waveform_hash(x),
        "waveform_hash": waveform_hash(x),
        "settings": asdict(settings),
        "smoothing_applied_samples": 0,
        "channel_status": channel_status,
        "automatic": automatic,
        "peak_idx": automatic["retained_idx"].copy(),
        "trough_overrides": np.empty((0, 3)),
        "preserved_troughs": np.empty((0, 3)),
        "manual_peak_edits": [],
        "manual_trough_edits": [],
        "uncertain_mask": ~masks["sample_valid"] & ~masks["artifact_mask"],
        "review_status": "automatic",
        **masks,
    }
    return recompute(belt, fs_hz)


def detect_preprocessed(data, settings=None):
    """Run the independent peak stage on a DBAT result or loaded result bundle.

    Parameters
    ----------
    data : PreprocessedRecording or dict
        Existing in-memory result, or load_preprocessing_result output. Uses
        only signals/smoothed, processed_fs_hz and processed-grid QC masks.
    settings : DetectorSettings, optional
        Respiratory detector settings; no preprocessing operation is performed.

    Returns
    -------
    dict
        Recording identity, explicit processed grid, source provenance and
        per-channel belt landmarks. Missing belt columns are simply absent.
    """
    if isinstance(data, dict):
        if data["respiratory_analysis_source"] != "signals/smoothed":
            raise ValueError("Respiratory detection requires signals/smoothed")
        values, fs = data["signals"]["smoothed"], data["processed_fs_hz"]
        time, channels, masks = data["processed_time_sec"], data["channels"], data["masks"]
        metadata = data["metadata"]
        identity, provenance, qc = metadata["identity"], metadata["processing"], metadata["qc"]
    else:
        values, fs = data.respiratory_signal, data.processed_fs_hz
        time, channels, masks = data.processed_time_sec, data.recording.channels, data.masks
        identity = recording_identity(data.recording)
        provenance, qc = data.provenance, data.qc
    values, time = np.asarray(values), np.asarray(time)
    if (
        values.ndim != 2
        or values.shape != (len(time), len(channels))
        or not len(time)
        or not np.isfinite(fs)
        or fs <= 0
        or not np.allclose(time, np.arange(len(time)) / fs, atol=1e-9, rtol=0)
    ):
        raise ValueError("Processed waveform, channels, fs and time grid disagree")
    for name in ("valid", "artifact", "interpolated"):
        if np.asarray(masks[name]).shape != values.shape:
            raise ValueError(f"Processed {name} mask does not match the waveform grid")
    result = {
        "algorithm_version": "dbat_respiratory_landmarks_v1",
        "identity": copy.deepcopy(identity),
        "fs_hz": float(fs),
        "time_sec": time.copy(),
        "source": "signals/smoothed",
        "preprocessing": copy.deepcopy(
            {
                key: provenance[key]
                for key in ("algorithm_version", "smoothing", "filtering", "resampling", "polarity")
                if key in provenance
            }
        ),
        "review_status": "automatic",
        "downstream_stale": True,
        "belts": {},
    }
    for column, name in enumerate(channels):
        if name not in {"Resp-Lungs", "Resp-Diaphragm"}:
            continue
        result["belts"][name] = detect_belt(
            values[:, column],
            fs,
            settings,
            valid=masks["valid"][:, column],
            artifact=masks["artifact"][:, column],
            interpolated=masks["interpolated"][:, column],
            source=f"signals/smoothed[{column}]",
            channel_status=qc["channel_ratings"][name],
        )
    if not result["belts"]:
        raise ValueError("No respiratory belt columns were supplied")
    return result
