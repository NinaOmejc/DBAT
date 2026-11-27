"""Run normal DBAT preprocessing/QC, verify the save, then review/edit peaks.

Run: python scripts/test_resp_peak_loading.py
Edit configs/default.toml for subjects, measurements, exclusions and QC settings.
peak_detection.source selects MATLAB loading or Python detection; both use the
same peak/trough editor. Reviewed / Save writes a separate landmark revision;
Cancel discards peak edits and continues to the next recording. Preprocessing
has already saved its normal immutable run and is never changed by peak edits.
MATLAB imports require the unchanged native sample grid (resample_freq=0).
Successful review also saves separate Stage 5a static references. Recalculate
from saved revisions without GUIs or raw data:
python scripts/test_resp_peak_loading.py --preprocessing FILE.h5 --landmarks FILE.h5
"""

import argparse
import json
from pathlib import Path
import tomllib

import numpy as np

from dbat.config import load_config
from dbat.features.reference import calculate_saved_references
from dbat.io.recording import load_dat_recording, load_preprocessing_result, source_sha256
from dbat.logging import configure_logging, logger
from dbat.peak_detection import detect_preprocessed, load_matlab_peaks, run_peak_check_gui
from dbat.preprocessing import run_preprocessing

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.toml"


def main(config_path=CONFIG_PATH, *, preprocessing_path=None, landmark_path=None, reference_output_dir=None):
    """Review configured recordings, or derive references from two saved revisions."""
    if preprocessing_path is not None or landmark_path is not None:
        if preprocessing_path is None or landmark_path is None:
            raise ValueError("Supply both --preprocessing and --landmarks")
        # Saved-only mode needs reference settings, not accessible raw input paths.
        with Path(config_path).open("rb") as stream:
            saved_config = tomllib.load(stream)
        configure_logging(saved_config["run"]["verbosity"])
        _, reference_path = calculate_saved_references(
            preprocessing_path, landmark_path, saved_config["reference"], reference_output_dir,
        )
        logger.info("Saved static references: %s", reference_path)
        return reference_path
    config = load_config(config_path)
    excluded = {tuple(pair) for pair in config["run"]["exclude_recordings"]}
    peaks_config = config["peak_detection"]
    if peaks_config["source"] == "matlab" and config["preprocessing"]["resampling"]["resample_freq"] not in (
        0,
        config["input"]["fs_hz"],
    ):
        raise ValueError("MATLAB landmarks require the native grid; set resample_freq=0")
    for subject in config["run"]["subjects"]:
        for measurement in config["run"]["measurements"]:
            if (subject, measurement) in excluded:
                continue
            recording = load_dat_recording(
                config,
                config["input"]["dat_file"] or None,
                subject=subject,
                measurement=measurement,
            )
            result, path = run_preprocessing(recording, config)
            if path is None:
                continue
            saved = load_preprocessing_result(path)
            np.testing.assert_array_equal(saved["preproc_signal"], result.preproc_signal)
            for name, mask in result.masks.items():
                np.testing.assert_array_equal(saved["masks"][name], mask)
            np.testing.assert_array_equal(saved["original_time_sec"], result.original_time_sec)
            np.testing.assert_array_equal(saved["processed_time_sec"], result.processed_time_sec)
            assert saved["channels"] == list(recording.channels)
            assert saved["original_fs_hz"] == result.original_fs_hz
            assert saved["processed_fs_hz"] == result.processed_fs_hz
            assert saved["metadata"]["qc"] == result.qc
            with (path.parent / "qc.json").open(encoding="utf-8") as stream:
                assert json.load(stream) == result.qc
            logger.info("Saved and checked: %s", path)
            logger.info(
                "Inspect: %s",
                path.parent / ("preview.png" if config["output"]["save_preview"] else "summary.json"),
            )
            if peaks_config["source"] == "matlab":
                cache = (
                    Path(peaks_config["matlab_results_dir"])
                    / recording.recording_id
                    / f"{recording.recording_id}_features.mat"
                )
                landmarks = load_matlab_peaks(cache, result)
            else:
                logger.info("Detecting respiratory peaks: %s", recording.recording_id)
                landmarks = detect_preprocessed(result)
            logger.info("Opening peak checker: %s (%s)", recording.recording_id, peaks_config["source"])
            landmarks["preprocessing"]["result_path"] = str(path)
            landmarks["preprocessing"]["result_sha256"] = source_sha256(path)
            _, accepted, peak_path = run_peak_check_gui(
                landmarks,
                peaks_config["output_dir"],
                window_sec=peaks_config["window_sec"],
                reviewer=config["qc"]["reviewer"],
            )
            if accepted:
                logger.info("Saved reviewed peaks: %s", peak_path)
                _, reference_path = calculate_saved_references(
                    path, peak_path, config["reference"], reference_output_dir,
                )
                logger.info("Saved static references: %s", reference_path)
            else:
                logger.info(
                    "Peak review cancelled: %s; preprocessing run remains saved", recording.recording_id
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--preprocessing", type=Path, help="Saved preprocessing HDF5 (no GUI)")
    parser.add_argument("--landmarks", type=Path, help="Matching reviewed landmark HDF5 (no GUI)")
    parser.add_argument("--reference-output-dir", type=Path, help="Override the separate references directory")
    args = parser.parse_args()
    main(args.config, preprocessing_path=args.preprocessing, landmark_path=args.landmarks,
         reference_output_dir=args.reference_output_dir)
