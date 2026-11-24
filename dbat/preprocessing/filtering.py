"""Optional SciPy Butterworth filtering across each finite native belt waveform."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .annotations import continuous_runs


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
    support = allowed.copy()
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
                support[start:end] = False
                run_info["skip_reason"] = "too_short_for_zero_phase_filter"
                continue
            pad = min(requested_pad, (count - 1) // 2)
            padded = np.pad(output[start:end], pad, mode="reflect")
            filtered = sosfiltfilt(sos, padded, padtype="odd", padlen=inner_pad)
            output[start:end] = filtered[pad : pad + count]
            support[start : start + pad] = False
            support[end - pad : end] = False
            if influence[start:end].any():
                influence[start:end] = True
            run_info["reflection_samples_each_end"] = pad
            details["applied"] = True
        details["support_policy"] = (
            "Filter all finite samples across QC/artifact flags; retain analysis masks "
            "and exclude reflection guards only at recording ends or genuine missing gaps"
        )
    return output, support, influence, provenance
