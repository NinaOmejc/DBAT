"""Shared-grid polyphase resampling with separate numerical and analysis support."""

from fractions import Fraction
from math import ceil, isclose

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import firwin, resample_poly


def resample_views(result, settings):
    """Resample preprocessed/smoothed values without erasing QC-excluded samples.

    Parameters
    ----------
    result : PreprocessingResult
        Native (samples, channels) numerical signals and independent Boolean masks.
    settings : dict
        Target frequency in Hz (0 disables), FIR half-width and Kaiser beta.

    Returns
    -------
    bundle : dict or None
        Shared-grid preprocessed/smoothed signals, time, fs and propagated masks.
    provenance : dict
        Filter parameters and explicit numerical/missing-data/support policies.

    Notes
    -----
    FIR calculations use actual finite values even when QC excludes them. Temporary
    linear filling is used only inside the filter for genuinely nonfinite inputs.
    Target samples bracketed by missing source data remain NaN; affected finite
    neighbours have false full-FIR support. Endpoint line extension retains useful
    display values while boundary/support masks exclude them from analysis.
    SciPy implements filter design and the resampling itself:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
    """
    native_fs = result.recording.fs_hz
    target_fs = settings["resample_freq"]
    provenance = {"requested_fs_hz": target_fs, "native_fs_hz": native_fs, "applied": False}
    if target_fs == 0 or target_fs == native_fs:
        provenance.update(
            fs_hz=native_fs, reason="disabled" if target_fs == 0 else "already_native"
        )
        return None, provenance
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
    n_samples, n_channels = result.original.shape
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
    masks = {
        "boundary": boundary,
        "reviewed": full_support(result.masks["reviewed"]),
        "eligible": full_support(result.masks["eligible"]),
        "artifact": any_influence(result.masks["artifact"]),
        "auto_candidate": any_influence(result.masks["auto_candidate"]),
        "uncertain": any_influence(result.masks["uncertain"]) | boundary,
        "quality_selected": result.masks["quality_selected"][left].copy(),
        "missing_influence": any_influence(~result.masks["finite"]),
    }
    bundle = {
        "fs_hz": effective_fs,
        "time_sec": np.arange(output_count) / effective_fs,
        "masks": masks,
    }
    index = np.arange(n_samples)
    for name, support_name, valid_name in (
        ("preprocessed", "preprocessed_support", "preprocessed_valid"),
        ("smoothed", "smoothed_support", "valid"),
    ):
        values = getattr(result, name)
        finite = np.isfinite(values)
        available = finite[left] & finite[right]
        # NaNs here represent real missing/absent source support, never QC status.
        resampled = np.full((output_count, n_channels), np.nan)
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
        numeric = available & np.isfinite(resampled)
        support = (
            full_support(result.masks[support_name] & finite & ~result.masks["artifact"]) & numeric
        )
        bundle[name] = resampled
        masks[support_name] = support
        masks[valid_name] = full_support(result.masks[valid_name]) & support
        dependence_name = f"{name}_interpolated"
        masks[dependence_name] = any_influence(result.masks[dependence_name]) & numeric
    masks["finite"] = np.isfinite(bundle["preprocessed"]) & np.isfinite(bundle["smoothed"])
    masks["interpolated"] = masks["preprocessed_interpolated"] | masks["smoothed_interpolated"]
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
        order="native preprocessed, native smoothed once, then shared-grid resampling",
        numerical_policy="Use actual finite values regardless of QC; retain unsupported numerical outputs",
        missing_policy="Temporary linear fill for FIR only; target samples bordering genuine gaps remain missing",
        support_policy="Full FIR footprint excludes padding, artifacts, missing data and unsupported processing",
        respiratory_analysis_source="resampled/signals/smoothed",
        counts={
            name: {key: int(mask[:, column].sum()) for key, mask in masks.items()}
            for column, name in enumerate(result.recording.channels)
        },
    )
    return bundle, provenance
