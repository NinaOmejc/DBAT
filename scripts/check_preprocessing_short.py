"""Run real DBAT preprocessing without the extensive synthetic checks.

Run: python scripts/check_preprocessing_short.py
Edit configs/default.toml to select recordings, QC review, methods and output.
Saves versioned HDF5, summary.json and an optional preview.png.
"""

from pathlib import Path

import numpy as np

from dbat.config import load_config
from dbat.io.recording import (
    load_dat_recording,
    load_preprocessing_result,
    load_qc_annotations,
    save_preprocessing_result,
    save_qc_annotations,
)
from dbat.logging import logger
from dbat.preprocessing.annotations import new_qc_annotations
from dbat.preprocessing.pipeline import prepare_recording, preprocess_recording

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.toml"


def main():
    """Process configured subject/measurement pairs and check saved arrays."""
    config = load_config(CONFIG_PATH)
    excluded = {tuple(pair) for pair in config["run"]["exclude_recordings"]}

    for subject in config["run"]["subjects"]:
        for measurement in config["run"]["measurements"]:
            if (subject, measurement) in excluded:
                continue

            # Load the original recording and any accepted previous QC.
            recording = load_dat_recording(
                config, config["input"]["dat_file"] or None,
                subject=subject, measurement=measurement,
            )
            qc_path = config["qc"]["annotation_files"].get(recording.recording_id)
            qc = (
                load_qc_annotations(qc_path, recording)
                if qc_path else new_qc_annotations(recording, config["qc"]["reviewer"])
            )

            # Prepare cleaning once; optionally inspect and accept QC in the GUI.
            prepared = prepare_recording(recording, config)
            if config["qc"]["open_gui"]:
                from dbat.preprocessing.preproc_gui import run_qc_gui

                qc = run_qc_gui(recording, config, qc, prepared=prepared)
                if qc is None:
                    logger.info("Cancelled %s; no result saved", recording.recording_id)
                    continue
                save_qc_annotations(qc, recording, config["paths"]["output_dir"])

            # Run the existing pipeline and save a new, versioned output.
            result = preprocess_recording(recording, config, qc, prepared=prepared)
            path = save_preprocessing_result(result, config)

            # Quick round-trip check (the original checker remains exhaustive).
            saved = load_preprocessing_result(path)
            for name in ("original", "preprocessed", "smoothed"):
                np.testing.assert_array_equal(saved["signals"][name], getattr(result, name))
            for name, mask in result.masks.items():
                np.testing.assert_array_equal(saved["masks"][name], mask)
            np.testing.assert_array_equal(saved["original_time_sec"], result.original_time_sec)
            np.testing.assert_array_equal(saved["processed_time_sec"], result.processed_time_sec)
            assert saved["channels"] == list(recording.channels)
            assert saved["original_fs_hz"] == result.original_fs_hz
            assert saved["processed_fs_hz"] == result.processed_fs_hz
            assert saved["metadata"]["qc"] == result.qc
            logger.info("Saved and checked: %s", path)
            logger.info("Inspect: %s", path.parent / (
                "preview.png" if config["output"]["save_preview"] else "summary.json"
            ))


if __name__ == "__main__":
    main()
