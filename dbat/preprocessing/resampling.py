"""Shared-grid polyphase resampling with separate numerical and analysis support."""

from fractions import Fraction
from math import ceil, isclose

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import firwin, resample_poly


def resample_preprocessed(values, masks, native_fs, settings):
    """Return the primary preprocessed grid and conservatively mapped masks.

    Parameters
    ----------
    values : ndarray, shape (native_samples, channels)
        Preprocessed native values. Smoothing happens once after this operation.
    masks : dict of ndarray
        Independent native QC, interpolation and preprocessing support masks.
    native_fs : float
        Original sampling rate in Hz.
    settings : dict
        Target frequency in Hz (0 disables), FIR half-width and Kaiser beta.

    Returns
    -------
    tuple
        Primary values, masks, sampling rate, zero-origin seconds, and provenance.
        Disabled/equal-rate requests return the original arrays without filtering.

    Notes
    -----
    Actual finite values enter the FIR regardless of review status. Genuine gaps
    are temporarily filled only inside the FIR, then restored on the target grid.
    Boundary/missing/artifact influence remains invalid but numerically available.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
    """
    target_fs = settings["resample_freq"]
    provenance = {"requested_fs_hz": target_fs, "native_fs_hz": native_fs, "applied": False}
    if target_fs == 0 or target_fs == native_fs:
        provenance.update(
            fs_hz=native_fs, reason="disabled" if target_fs == 0 else "already_native"
        )
        return values, masks, native_fs, np.arange(len(values)) / native_fs, provenance
    ratio = (Fraction(str(target_fs)) / Fraction(str(native_fs))).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    effective_fs = native_fs * up / down
    if up == down or max(up, down) > 10000 or not isclose(effective_fs, target_fs, rel_tol=1e-10):
        raise ValueError(
            "Resampling needs a rate ratio representable with up/down factors <= 10000"
        )
    half_length = settings["filter_half_width"] * max(up, down)
    coefficients = firwin(
        2 * half_length + 1, 1 / max(up, down), window=("kaiser", settings["kaiser_beta"])
    )
    radius = ceil(half_length / up)
    n_samples, n_channels = values.shape
    output_count = (n_samples * up + down - 1) // down
    numerator = np.arange(output_count) * down
    left = np.minimum(numerator // up, n_samples - 1)
    right = np.minimum((numerator + up - 1) // up, n_samples - 1)

    def full_support(mask):
        """Require every native sample in the conservative FIR footprint."""
        eroded = minimum_filter1d(
            mask.astype(np.uint8), 2 * radius + 1, axis=0, mode="constant", cval=0
        )
        return eroded[left].astype(bool)

    def any_influence(mask):
        """Retain any flagged contribution within the same FIR footprint."""
        expanded = maximum_filter1d(
            mask.astype(np.uint8), 2 * radius + 1, axis=0, mode="constant", cval=0
        )
        return expanded[left].astype(bool)

    boundary = np.broadcast_to(
        ~full_support(np.ones((n_samples, 1), dtype=bool)), (output_count, n_channels)
    ).copy()
    native_masks = masks
    masks = {
        "boundary": boundary,
        "reviewed": full_support(native_masks["reviewed"]),
        "eligible": full_support(native_masks["eligible"]),
        "artifact": any_influence(native_masks["artifact"]),
        "auto_candidate": any_influence(native_masks["auto_candidate"]),
        "unresolved_candidate": any_influence(native_masks["unresolved_candidate"]),
        "uncertain": any_influence(native_masks["uncertain"]) | boundary,
        "quality_selected": native_masks["quality_selected"][left].copy(),
        "missing_influence": any_influence(~native_masks["finite"]),
    }
    finite = np.isfinite(values)
    available = finite[left] & finite[right]
    resampled = np.full((output_count, n_channels), np.nan)
    index = np.arange(n_samples)
    for column in range(n_channels):
        observed = finite[:, column]
        if not observed.any():
            continue
        numerical = values[:, column]
        if not observed.all():
            numerical = np.interp(index, index[observed], numerical[observed])
        resampled[:, column] = resample_poly(
            numerical, up, down, window=coefficients, padtype="line"
        )
    resampled[~available] = np.nan
    masks["finite"] = np.isfinite(resampled)
    masks["preprocessed_support"] = (
        full_support(native_masks["preprocessed_support"] & finite & ~native_masks["artifact"])
        & masks["finite"]
    )
    masks["preprocessed_valid"] = (
        full_support(native_masks["preprocessed_valid"]) & masks["preprocessed_support"]
    )
    masks["preprocessed_interpolated"] = (
        any_influence(native_masks["preprocessed_interpolated"]) & masks["finite"]
    )
    masks["interpolated"] = any_influence(native_masks["interpolated"]) & masks["finite"]
    provenance.update(
        applied=True,
        method="scipy.signal.resample_poly",
        fs_hz=effective_fs,
        up=up,
        down=down,
        n_samples=output_count,
        fir_numtaps=len(coefficients),
        fir_cutoff_hz=min(native_fs, effective_fs) / 2,
        filter_half_width=settings["filter_half_width"],
        kaiser_beta=settings["kaiser_beta"],
        native_support_radius_samples=radius,
        padtype="line",
        order="native detrending/filtering, resample preprocessed once, then one processed-grid smoothing pass",
        numerical_policy="Use actual finite values regardless of QC; retain unsupported numerical outputs",
        missing_policy="Temporary linear fill for FIR only; target samples bordering genuine gaps remain missing",
        support_policy="Full FIR footprint excludes padding, artifacts, missing data and unsupported processing",
        respiratory_analysis_source="preproc_signal",
    )
    return resampled, masks, effective_fs, np.arange(output_count) / effective_fs, provenance
