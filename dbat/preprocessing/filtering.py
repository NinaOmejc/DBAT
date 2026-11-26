"""Optional SciPy Butterworth filtering across each finite native belt waveform."""

from copy import deepcopy

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..config import BELTS
from .annotations import continuous_runs
from .cleaning import preprocess_bp, preprocess_ecg
from .detrending import detrend_belt, detrending_parameters


def filtering_support(allowed, filters):
    """Apply recorded finite-run reflection/short-run guards without filtering values."""
    support = allowed.copy()
    for details in filters.values():
        for run in details.get("runs", []):
            start, end = run["start_sample"], run["end_sample"]
            if "skip_reason" in run:
                support[start:end] = False
            else:
                pad = run["reflection_samples_each_end"]
                support[start : start + pad] = False
                support[end - pad : end] = False
    return support


def filter_belt(values, allowed, interpolated, fs_hz, settings):
    """Filter a native belt continuously across QC and artifact flags.

    Parameters
    ----------
    values : ndarray, shape (n_samples,)
        Numerical belt values; the input is never modified.
    allowed, interpolated : ndarray of bool
        Analysis-support mask and existing interpolation influence. These masks
        do not split or remove numerical samples before filtering.
    fs_hz : float
        Native sampling rate in Hz.
    settings : dict
        Separate highpass/lowpass enabled, cutoff_hz and edge_pad_sec settings.

    Returns
    -------
    tuple
        Values, support, interpolation influence, and actual filter/run provenance.
        Short runs retain their input with false support. Reflection guards stay
        numerical but unsupported. IIR interpolation influence covers the whole
        finite waveform. Only genuine nonfinite gaps create separate filter runs.

    Notes
    -----
    Uses SciPy's established SOS implementation, not a custom IIR implementation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
    Outer even reflection and its cap follow the MIT-licensed MATLAB reference:
    https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/preprocess_data.m
    QC flags do not create reflection boundaries. Genuine nonfinite gaps remain
    missing and split filtering into finite runs. The inner
    odd extension is explicitly 3 * order = 12 samples (MATLAB filtfilt convention).
    Forward/backward application squares the single-pass magnitude response.
    """
    output = values.copy()
    influence = interpolated.copy()
    provenance = {}
    order = 4
    for name in ("highpass", "lowpass"):
        options = settings[name]
        details = {**options, "applied": False, "order": order, "fs_hz": fs_hz}
        provenance[name] = details
        if not options["enabled"]:
            continue
        sos = butter(order, options["cutoff_hz"], btype=name, fs=fs_hz, output="sos")
        inner_pad = 3 * order
        requested_pad = max(inner_pad, int(np.floor(options["edge_pad_sec"] * fs_hz + 0.5)))
        details.update(
            method="scipy.signal.sosfiltfilt",
            sos=sos.tolist(),
            outer_pad="even reflection excluding endpoints",
            inner_pad="odd",
            inner_pad_samples=inner_pad,
            runs=[],
        )
        for start, end in continuous_runs(np.isfinite(values)):
            count = end - start
            run_info = {"start_sample": int(start), "end_sample": int(end)}
            details["runs"].append(run_info)
            if count <= inner_pad:
                run_info["skip_reason"] = "too_short_for_zero_phase_filter"
                continue
            pad = min(requested_pad, (count - 1) // 2)
            padded = np.pad(output[start:end], pad, mode="reflect")
            filtered = sosfiltfilt(sos, padded, padtype="odd", padlen=inner_pad)
            output[start:end] = filtered[pad : pad + count]
            if influence[start:end].any():
                influence[start:end] = True
            run_info["reflection_samples_each_end"] = pad
            details["applied"] = True
        details["support_policy"] = (
            "Filter all finite samples across QC/artifact flags; retain analysis masks "
            "and exclude reflection guards only at recording ends or genuine missing gaps"
        )
    return output, filtering_support(allowed, provenance), influence, provenance


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
        skipped detrend runs, and the unsaved detrended vector. Detrending splits
        at gaps/artifacts; filtering follows each finite waveform across QC flags. Ratings do not split either.
    """
    output, allowed, support, influence, skips = detrend_belt(
        values, artifact, interpolated, detrend
    )
    detrended = output
    output, filter_support, influence, filter_details = filter_belt(
        output, allowed, influence, fs_hz, filtering
    )
    return output, support & filter_support, influence, filter_details, skips, detrended


def detrend_and_filter_recording(
    corrected, recording, settings, masks, qc=None, *, previous=None, changed=None
):
    """Apply native belt conditioning in the original per-channel order.

    Parameters
    ----------
    corrected : ndarray, shape (native_samples, channels)
        Corrected, oriented signals; left unchanged.
    recording : Recording
        Native channel order and sampling frequency in Hz.
    settings : dict
        Existing preprocessing settings, including enabled and selected channels.
    masks : dict of ndarray
        Native masks; preprocessing support/influence/validity are added in place.
    qc : dict, optional
        Accepted or private preview per-belt baseline choices, overriding config.
    previous : PreprocessedRecording, optional
        Preview cache; unchanged channels reuse native values and stage diagnostics.
    changed : set of str, optional
        Channels whose numerical signatures changed, checked by the caller.

    Returns
    -------
    tuple
        Preprocessed values and the original native-stage diagnostics. Each
        selected belt is detrended, high-pass filtered and low-pass filtered
        before advancing to the next channel. Other channels pass through.
    """
    # Previous native arrays are private preview caches, never saved waveforms.
    per_channel = {}
    filtering = {
        name: {**options, "enabled": options["enabled"] and settings["enabled"]}
        for name, options in settings["filtering"].items()
    }
    preprocessed = corrected.copy()
    detrended = corrected.copy()
    masks["preprocessed_support"] = masks["finite"] & ~masks["artifact"]
    masks["preprocessed_interpolated"] = masks["interpolated"].copy()
    processors, filter_details, processing_skips = {}, {}, {}
    for column, name in enumerate(recording.channels):
        if name in BELTS and name in settings["channels"]:
            parameters = detrending_parameters(settings, recording.fs_hz, name, qc)
            per_channel[name] = parameters
            if previous is not None and name not in changed:
                per_channel[name] = deepcopy(previous.provenance["detrend"]["per_channel"][name])
                preprocessed[:, column] = previous.native_preprocessed[:, column]
                detrended[:, column] = previous.native_detrended[:, column]
                filter_details[name] = previous.provenance["filtering"][name]
                processing_skips[name] = deepcopy(previous.provenance["processing_skips"][name])
                native = previous.native_conditioning_masks
                masks["preprocessed_support"][:, column] = (
                    filtering_support(masks["preprocessed_support"][:, column], filter_details[name])
                    if parameters["method"] == "none"
                    else native["preprocessed_support"][:, column]
                )
                masks["preprocessed_interpolated"][:, column] = native["preprocessed_interpolated"][:, column]
                processors[name] = "belt_conditioning"
                continue
            (
                preprocessed[:, column],
                masks["preprocessed_support"][:, column],
                masks["preprocessed_interpolated"][:, column],
                filter_details[name],
                skips,
                detrended[:, column],
            ) = preprocess_belt(
                corrected[:, column],
                masks["artifact"][:, column],
                masks["interpolated"][:, column],
                recording.fs_hz,
                parameters,
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
    methods = {parameters["method"] for parameters in per_channel.values()}
    detrend = {
        key: value for key, value in next(iter(per_channel.values()), {}).items()
        if key != "runs"
    }
    detrend["method"] = next(iter(methods)) if len(methods) == 1 else "mixed"
    if not methods:
        detrend["method"] = "none"
    detrend["per_channel"] = per_channel
    return preprocessed, {
        "channel_processors": processors,
        "detrend": detrend,
        "filtering": filter_details,
        "processing_skips": processing_skips,
    }, detrended
