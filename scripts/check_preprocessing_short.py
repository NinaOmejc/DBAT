"""Run real DBAT preprocessing without the extensive synthetic checks.

Run: python scripts/check_preprocessing_short.py
Edit configs/default.toml to select recordings, QC review, methods and output.
Reuses compatible runs or saves versioned HDF5, summary.json and an optional preview.
"""

from pathlib import Path

import numpy as np

from dbat.config import load_config
from dbat.io.recording import load_dat_recording, load_preprocessing_result
from dbat.logging import logger
from dbat.preprocessing import run_preprocessing

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.toml"


def main():
    """Process configured subject/measurement pairs and check saved arrays."""
    config = load_config(CONFIG_PATH)
    excluded = {tuple(pair) for pair in config["run"]["exclude_recordings"]}

    for subject in config["run"]["subjects"]:
        for measurement in config["run"]["measurements"]:
            if (subject, measurement) in excluded:
                continue

            # Load the source once, then reuse or process through the shared workflow.
            recording = load_dat_recording(
                config, config["input"]["dat_file"] or None,
                subject=subject, measurement=measurement,
            )
            result, path, reused = run_preprocessing(recording, config)
            if path is None:
                continue

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
            logger.info("%s: %s", "Reused" if reused else "Saved and checked", path)
            logger.info("Inspect: %s", path.parent / (
                "preview.png" if config["output"]["save_preview"] else "summary.json"
            ))


if __name__ == "__main__":
    main()
