"""Manually compare one recording's DBAT preprocessing and saved MATLAB peaks.

Run: python scripts/test_resp_peak_loading.py
Edit subject/measurement in the final main() call. Uses configs/default.toml,
reuses available DBAT QC, and preprocesses in memory without opening QC or saving.
Keep resample_freq=0: historical native indices are rejected on a changed grid.
The overlay is a comparison, not approval/import of landmarks for DBAT analysis.

MATLAB cache layout and one-based index/time conventions:
https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/load_or_extract_respiratory_cycles.m
https://github.com/NinaOmejc/magma-labeling/blob/main/src/feature_extraction/recompute_respiration_breath_fields.m
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from dbat.config import BELTS, load_config
from dbat.io.recording import load_dat_recording, load_latest_qc_annotations
from dbat.logging import logger
from dbat.preprocessing.pipeline import preprocess_recording

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
MATLAB_RESULTS_DIR = Path("D:/Projects/MAGMA/data_analysis/statistical_labeling")


def load_matlab_peaks(path, recording):
    """Read standard MATLAB feature-cache peaks on the recording's native grid.

    Returns a dict per belt with zero-based peak_idx, seconds, saved detection
    waveform and source-relative polarity. Validates cache identity/grid, vector
    lengths, integer bounds and index/time agreement. These caches have no raw
    source-file hash; matching metadata alone cannot establish waveform identity.
    """
    saved = loadmat(path, simplify_cells=True, variable_names=["resp_cycles", "feature_cache_meta"])
    if not {"resp_cycles", "feature_cache_meta"} <= saved.keys():
        raise ValueError("Expected a MATLAB *_features.mat cache with resp_cycles and feature_cache_meta")
    meta = saved["feature_cache_meta"]
    for key, expected in (
        ("subject", recording.subject),
        ("measurement", recording.measurement),
        ("fs", recording.fs_hz),
        ("n_samples", len(recording.samples)),
    ):
        if meta[key] != expected:
            raise ValueError(f"MATLAB {key}={meta[key]} differs from recording {expected}: {path}")
    columns = [str(name).replace("\u2082", "2") for name in np.atleast_1d(meta["data_columns"])]
    if columns != list(recording.channels):
        raise ValueError("MATLAB channel order differs from the recording")
    peaks = {}
    for channel, key in zip(BELTS, ("lungs", "diaph"), strict=True):
        belt = saved["resp_cycles"][key]
        waveform = np.asarray(belt["x0"], dtype=float).reshape(-1)
        indices = np.asarray(belt["peak_idx"], dtype=float).reshape(-1)
        times = np.asarray(belt["peak_t"], dtype=float).reshape(-1)
        if len(waveform) != len(recording.samples):
            raise ValueError(f"{channel}: MATLAB waveform has a different sample count")
        if (
            not np.isfinite(indices).all()
            or np.any(indices != np.floor(indices))
            or np.any(indices < 1)
            or np.any(indices > len(waveform))
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError(f"{channel}: MATLAB peaks must be ordered, unique one-based native indices")
        indices = indices.astype(np.int64) - 1
        if times.shape != indices.shape or not np.allclose(
            times, indices / recording.fs_hz, rtol=0, atol=1e-9
        ):
            raise ValueError(f"{channel}: MATLAB peak times disagree with its indices/fs")
        if not np.isfinite(waveform[indices]).all():
            raise ValueError(f"{channel}: MATLAB peak values lie on missing samples")
        # Cache v9 has no thoracic flip flag; MATLAB treats it as false.
        relative_flip = (
            meta.get("thoracic_polarity_flipped", False)
            if key == "lungs"
            else meta["diaphragm_polarity_flipped"]
        )
        peaks[channel] = {
            "peak_idx": indices,
            "peak_t": times,
            "matlab_waveform": waveform,
            "matlab_polarity_flipped": bool(relative_flip) ^ bool(meta["absolute_polarity_flipped"]),
        }
    return peaks


def main(
    subject=100, measurement=1, *, config_path=CONFIG_PATH, matlab_results_dir=MATLAB_RESULTS_DIR, show=True
):
    """Preprocess one recording and overlay saved MATLAB peaks; return result, peaks.

    Existing QC remains authoritative; absent/unaccepted QC stays provisional.
    Markers use the saved MATLAB waveform so changed DBAT amplitudes or polarity
    remain visible. show=False supports a headless manual check without writing.
    """
    config = load_config(config_path)
    recording = load_dat_recording(
        config, config["input"]["dat_file"] or None, subject=subject, measurement=measurement
    )
    qc = load_latest_qc_annotations(recording, config)
    result = preprocess_recording(recording, config, qc, provisional=True)
    result.require_native_grid()
    cache = Path(matlab_results_dir) / recording.recording_id / f"{recording.recording_id}_features.mat"
    peaks = load_matlab_peaks(cache, recording)
    logger.info("Loaded MATLAB peaks: %s", cache)
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 7))
    figure.suptitle(f"{recording.recording_id}: historical MATLAB peaks vs DBAT (comparison only)")
    for axis, channel in zip(axes, BELTS, strict=True):
        column = recording.channels.index(channel)
        saved = peaks[channel]
        indices, reference = saved["peak_idx"], saved["matlab_waveform"]
        current = result.preproc_signal[:, column]
        saved["dbat_valid"] = result.masks["valid"][indices, column]
        saved["waveform_matches"] = bool(
            np.allclose(current, reference, rtol=1e-7, atol=1e-9, equal_nan=True)
        )
        flipped = result.provenance["polarity"]["applied_to"] == channel
        if flipped != saved["matlab_polarity_flipped"]:
            logger.warning("%s: MATLAB and DBAT polarity differ; peaks remain on the MATLAB trace", channel)
        if not saved["waveform_matches"]:
            logger.warning(
                "%s: preprocessing changed the waveform; historical peaks need review before reuse", channel
            )
        logger.info(
            "%s: %s MATLAB peaks; %s QC-valid in DBAT; waveform match=%s",
            channel,
            len(indices),
            int(saved["dbat_valid"].sum()),
            saved["waveform_matches"],
        )
        axis.plot(
            result.processed_time_sec, current, color="tab:blue", lw=1.1, zorder=3, label="DBAT preprocessed"
        )
        axis.plot(
            recording.time_sec,
            reference,
            color="0.5",
            ls="--",
            lw=0.8,
            zorder=2,
            label="MATLAB detection waveform",
        )
        axis.scatter(
            saved["peak_t"],
            reference[indices],
            s=22,
            facecolors="none",
            edgecolors="crimson",
            zorder=4,
            label="Saved MATLAB peaks",
        )
        axis.set_ylabel(channel)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (seconds); zoom/pan with the Matplotlib toolbar")
    axes[-1].set_xlim(0, recording.time_sec[-1])
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    if show:
        plt.show()
    return result, peaks


if __name__ == "__main__":
    main(subject=100, measurement=1)
