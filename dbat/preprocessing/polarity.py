"""Relative belt diagnostics and explicit, source-relative orientation choices."""

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..config import BELTS

POLARITY_LABELS = ("None", "Invert Resp-Lungs", "Invert Resp-Diaphragm")
POLARITY_CHOICES = (None, *BELTS)
MATLAB_SOURCE = "https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/correct_respiratory_belt_polarity.m"


def assess_relative_belt_polarity(values, channels, fs_hz, reliable, enabled):
    """Assess persistent negative belt correlation with MATLAB's window definitions.

    Parameters
    ----------
    values : ndarray, shape (n_samples, n_channels)
        Corrected native signals in source orientation; never modified.
    channels : sequence of str
        Native column order.
    fs_hz : float
        Native sampling rate in Hz.
    reliable : ndarray of bool
        Numerical reliability, independent of manual channel-usability ratings.
    enabled : bool
        False returns an explicit disabled assessment.

    Returns
    -------
    dict
        Correlation diagnostics and warning only; never an automatic flip.

    Notes
    -----
    Adapted from the MIT-licensed MATLAB method:
    https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/correct_respiratory_belt_polarity.m
    SOS filtering uses SciPy; MATLAB numerical equivalence has not been measured.
    """
    result = {
        "algorithm_version": "dbat_relative_polarity_v1",
        "source": MATLAB_SOURCE,
        "source_sha256": "d3549016ceb1d748bb145906327119993b7a2ce8e8e1734b5030434e6df42986",
        "orientation_basis": "corrected_source_orientation_before_manual_flip",
        "checked": bool(enabled),
        "status": "not_assessed",
        "reason": "disabled",
        "polarity_warning": False,
        "n_windows": 0,
        "n_informative_windows": 0,
        "median_window_corr": None,
        "negative_window_fraction": None,
        "informative_window_fraction": None,
        "windows": [],
        "parameters": {
            "band_hz": [0.05, 0.60],
            "butterworth_order": 4,
            "zero_phase": True,
            "window_sec": 60,
            "step_sec": 30,
            "minimum_data_sec": 30,
            "minimum_paired_fraction": 0.80,
            "informative_abs_r": 0.50,
            "minimum_informative_windows": 3,
            "warning_median_r": -0.65,
            "warning_negative_fraction": 0.80,
            "warning_informative_fraction": 0.60,
        },
    }
    if not enabled:
        return result
    result.update(status="insufficient_reliable_data", reason="two_reliable_belts_required")
    if not set(BELTS) <= set(channels) or fs_hz / 2 <= 0.60:
        return result
    columns = [channels.index(name) for name in BELTS]
    paired = reliable[:, columns].all(axis=1) & np.isfinite(values[:, columns]).all(axis=1)
    minimum = max(3, math.ceil(30 * fs_hz))
    if paired.sum() < minimum:
        return result
    sos = butter(4, [0.05, 0.60], btype="bandpass", fs=fs_hz, output="sos")
    # Order-based transient length: three times the bandpass filter order.
    # https://www.mathworks.com/help/signal/ref/filtfilt.html
    padlen = 3 * (2 * len(sos))
    result["parameters"].update(padtype="odd", padlen_samples=padlen)
    if len(values) <= padlen:
        return result
    band = []
    index = np.arange(len(values))
    for column in columns:
        signal = values[:, column]
        valid = reliable[:, column] & np.isfinite(signal)
        if valid.sum() < minimum:
            return result
        filled = np.interp(index, index[valid], signal[valid])
        filtered = sosfiltfilt(sos, filled, padtype="odd", padlen=padlen)
        filtered[~valid] = np.nan
        band.append(filtered)
    width, step = max(1, round(60 * fs_hz)), max(1, round(30 * fs_hz))
    windows = range(0, len(values) - width + 1, step) if len(values) >= width else [0]
    correlations = []
    for start in windows:
        end = min(start + width, len(values))
        valid = np.isfinite(band[0][start:end]) & np.isfinite(band[1][start:end])
        if valid.sum() < math.ceil(0.80 * (end - start)):
            continue
        vectors = [signal[start:end][valid] for signal in band]
        meaningful = all(
            np.std(vector, ddof=1)
            > max(1e-12, np.sqrt(np.finfo(float).eps) * max(np.max(np.abs(vector)), 1e-6))
            for vector in vectors
        )
        if not meaningful:
            continue
        correlation = float(np.corrcoef(vectors)[0, 1])
        if not np.isfinite(correlation):
            continue
        correlations.append(correlation)
        result["windows"].append(
            {"start_sample": int(start), "end_sample": int(end), "r": correlation}
        )
    result["n_windows"] = len(correlations)
    informative = np.asarray(correlations)[np.abs(correlations) >= 0.50]
    result["n_informative_windows"] = len(informative)
    if correlations:
        result["informative_window_fraction"] = len(informative) / len(correlations)
    if informative.size:
        result["median_window_corr"] = float(np.median(informative))
        result["negative_window_fraction"] = float(np.mean(informative <= -0.50))
    if len(informative) < 3:
        result["reason"] = "too_few_informative_windows"
        return result
    warning = (
        result["median_window_corr"] <= -0.65
        and result["negative_window_fraction"] >= 0.80
        and result["informative_window_fraction"] >= 0.60
    )
    result.update(
        status="assessed",
        reason="persistent_negative_correlation" if warning else "criteria_not_met",
        polarity_warning=warning,
    )
    return result


def saved_polarity_decision(qc, prepared):
    """Return a source/cleaning-compatible accepted choice, or reject a stale choice."""
    decision = qc.get("polarity")
    if decision is None:
        return None
    if decision["preparation_signature"] != prepared["signature"]:
        raise ValueError(
            "Saved polarity decision uses different cleaning settings; review polarity again in the GUI"
        )
    return decision


def apply_polarity(values, channels, choice):
    """Orient a fresh corrected copy relative to its source, avoiding double flips."""
    if choice not in POLARITY_CHOICES:
        raise ValueError("Polarity choice must be None, Resp-Lungs or Resp-Diaphragm")
    oriented = values.copy()
    if choice is not None:
        oriented[:, channels.index(choice)] *= -1
    return oriented


def assessment_text(assessment):
    """Format a neutral or assessed diagnostic; only complete warning criteria are red."""
    if assessment["status"] == "not_assessed":
        return "Not assessed"
    if assessment["status"] != "assessed":
        return "Insufficient reliable data"
    text = (
        f"Median r: {assessment['median_window_corr']:+.2f}\n"
        f"Negative windows: {assessment['negative_window_fraction']:.0%} | "
        f"Informative: {assessment['informative_window_fraction']:.0%}"
    )
    return text + (
        "\nCheck polarity; paradoxical motion is also possible"
        if assessment["polarity_warning"]
        else ""
    )


def orient_prepared(prepared, qc, channels):
    """Validate the saved polarity decision and orient a fresh corrected copy.

    Returns oriented samples and the unchanged decision (or None). Warnings
    still require an explicit reviewer choice or acknowledgement.
    """
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
    corrected = apply_polarity(prepared["corrected"], channels, choice)
    return corrected, decision
