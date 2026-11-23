"""Optional SciPy polyphase resampling on a shared grid, retaining native QC."""

from fractions import Fraction
from math import ceil, isclose

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import firwin, resample_poly


def resample_views(result, settings):
    """Resample all derived channels together; return a bundle and provenance.

    Native source, time and annotations are unchanged. FIR support is eroded
    conservatively around missing values, artifacts and unsupported input samples;
    no output influenced by these samples or padding is marked supported.
    Quality-valid masks additionally require full reviewed, usable input support.

    Parameters
    ----------
    result : PreprocessingResult
        Native views and masks, each shaped (n_samples, n_channels).
    settings : dict
        Target frequency in Hz (0 disables), FIR half-width and Kaiser beta.

    Returns
    -------
    bundle : dict or None
        Shared-grid views, time in seconds, fs in Hz and boolean support/valid masks.
    provenance : dict
        Applied ratio, filter parameters, support rules and per-channel counts.

    Notes
    -----
    SciPy supplies both filter design and polyphase resampling:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
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
    n_samples = result.recording.samples.shape[0]
    output_count = (n_samples * up + down - 1) // down
    source_indices = np.minimum(np.arange(output_count) * down // up, n_samples - 1)

    def full_support(mask):
        """Map full FIR input support to the shared output grid."""
        # All native contributors must be supported, including fractional centers.
        eroded = minimum_filter1d(
            mask.astype(np.uint8), 2 * radius + 1, axis=0, mode="constant", cval=0
        )
        return eroded[source_indices].astype(bool)

    bundle = {
        "fs_hz": effective_fs,
        "time_sec": np.arange(output_count) / effective_fs,
        "masks": {},
    }
    for name, support_name, valid_name in (
        ("conditioned", "conditioning_support", "conditioned_valid"),
        ("respiratory_view", "respiratory_support", "valid"),
    ):
        values = getattr(result, name)
        native_support = (
            result.masks[support_name] & np.isfinite(values) & ~result.masks["artifact"]
        )
        support = full_support(native_support)
        # Zeros are only filter-buffer placeholders: every affected output is NaN.
        resampled = resample_poly(
            np.where(native_support, values, 0.0),
            up,
            down,
            axis=0,
            window=coefficients,
            padtype="constant",
            cval=0.0,
        )
        support &= np.isfinite(resampled)
        resampled[~support] = np.nan
        bundle[name] = resampled
        bundle["masks"][support_name] = support
        bundle["masks"][valid_name] = full_support(result.masks[valid_name]) & support
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
        order="native signal-specific conditioning, then shared resampling",
        support_policy="Full FIR footprint required; padding/artifacts/missing/unsupported input excluded",
        counts={
            name: {key: int(mask[:, column].sum()) for key, mask in bundle["masks"].items()}
            for column, name in enumerate(result.recording.channels)
        },
    )
    return bundle, provenance
