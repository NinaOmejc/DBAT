"""Manually inspect real recordings selected in one grouped DBAT configuration.

Run directly in PyCharm or use:
    python scripts/check_preprocessing.py --config configs/default.toml
Processes every selected subject/measurement pair, skipping explicit exclusions.
Writes versioned HDF5/JSON results and optional previews, then checks each round trip.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from dbat.config import BELTS, RESULT_SCHEMA, load_config
from dbat.io.recording import (
    load_dat_recording,
    load_qc_annotations,
    save_preprocessing_result,
    save_qc_annotations,
    source_sha256,
)
from dbat.logging import configure_logging, logger
from dbat.preprocessing.annotations import (
    new_qc_annotations,
    utc_now,
    validate_qc_annotations,
)
from dbat.preprocessing.pipeline import prepare_recording, preprocess_recording

REPOSITORY = Path(__file__).resolve().parents[1]


def verify_saved_result(path, result):
    """Reopen output and check native originals, primary grids, masks and metadata."""
    recording = result.recording
    logger.info("  Verifying saved result for %s...", recording.recording_id)
    with (path.parent / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    expected_arrays = {
        "signals/original": result.original,
        "signals/preprocessed": result.preprocessed,
        "signals/smoothed": result.smoothed,
        "original_time_sec": result.original_time_sec,
        "processed_time_sec": result.processed_time_sec,
        **{f"masks/{name}": mask for name, mask in result.masks.items()},
    }
    with h5py.File(path, "r") as handle:
        metadata = json.loads(handle["metadata_json"].asstr()[()])
        if (
            metadata != summary
            or metadata["schema_version"] != RESULT_SCHEMA
            or handle.attrs["schema_version"] != RESULT_SCHEMA
        ):
            raise ValueError("HDF5 and JSON metadata/schema disagree")
        if (
            handle.attrs["original_fs_hz"] != result.original_fs_hz
            or handle.attrs["processed_fs_hz"] != result.processed_fs_hz
            or handle["channels"].asstr()[:].tolist() != list(recording.channels)
        ):
            raise ValueError("Sampling rates or channel order changed during saving")
        if (
            set(handle["signals"]) != {"original", "preprocessed", "smoothed"}
            or "resampled" in handle
        ):
            raise ValueError(
                "Expected exactly three canonical signals and no duplicate resampled bundle"
            )
        for name, expected in expected_arrays.items():
            saved = handle[name][:]
            if (
                saved.dtype != expected.dtype
                or saved.shape != expected.shape
                or not np.array_equal(saved, expected, equal_nan=True)
            ):
                raise ValueError(f"Round-trip data mismatch: {name}")
        for name in ("original", "preprocessed", "smoothed"):
            expected_fs = result.original_fs_hz if name == "original" else result.processed_fs_hz
            expected_axis = "original_time_sec" if name == "original" else "processed_time_sec"
            if (
                handle[f"signals/{name}"].attrs["fs_hz"] != expected_fs
                or handle[f"signals/{name}"].attrs["time_axis"] != expected_axis
            ):
                raise ValueError(f"Signal sampling metadata disagrees: {name}")
        if metadata["qc"] != result.qc:
            raise ValueError("Saved QC annotations differ from the accepted input")
        validate_qc_annotations(metadata["qc"], recording)
    if not np.array_equal(result.original, recording.samples, equal_nan=True):
        raise ValueError("Original samples changed")
    n_samples = len(result.preprocessed)
    if result.smoothed.shape != (n_samples, len(recording.channels)):
        raise ValueError("Primary processed signal shapes disagree")
    for times, fs_hz, count in (
        (result.original_time_sec, result.original_fs_hz, len(result.original)),
        (result.processed_time_sec, result.processed_fs_hz, n_samples),
    ):
        if not np.array_equal(times, np.arange(count) / fs_hz):
            raise ValueError("Signal time grid disagrees with its zero origin and sampling rate")
    rate = result.provenance["resampling"]
    if rate["applied"]:
        expected_count = (len(result.original) * rate["up"] + rate["down"] - 1) // rate["down"]
        if n_samples != expected_count:
            raise ValueError("Primary processed sample count disagrees with the resampling ratio")
        numerator = np.arange(n_samples) * rate["down"]
        left = np.minimum(numerator // rate["up"], len(result.original) - 1)
        right = np.minimum((numerator + rate["up"] - 1) // rate["up"], len(result.original) - 1)
        available = np.isfinite(result.original[left]) & np.isfinite(result.original[right])
    else:
        result.require_native_grid()
        available = np.isfinite(result.original)
    for values in (result.preprocessed, result.smoothed):
        if np.any(available & ~np.isfinite(values)) or np.any(~available & np.isfinite(values)):
            raise ValueError("Numerical availability disagrees with genuine source support")
    for name, mask in result.masks.items():
        if mask.dtype != bool or mask.shape != result.smoothed.shape:
            raise ValueError(f"Mask is not Boolean on the primary processed grid: {name}")
    if np.any(
        result.masks["valid"]
        & (~result.masks["reviewed"] | result.masks["artifact"] | ~result.masks["smoothed_support"])
    ):
        raise ValueError("Processing accepted unreviewed, artifact or unsupported samples")
    if rate["applied"] and np.any(
        result.masks["valid"] & (result.masks["boundary"] | result.masks["missing_influence"])
    ):
        raise ValueError("Processing accepted contaminated FIR support")
    if (
        result.respiratory_signal is not result.smoothed
        or result.provenance["respiratory_analysis_fs_hz"] != result.processed_fs_hz
    ):
        raise ValueError("Respiratory analysis must use smoothed and its processed sampling rate")
    for column, name in enumerate(recording.channels):
        if (
            name not in BELTS
            or name not in result.provenance["selected_channels"]
            or result.provenance["smoothing"]["method"] == "none"
        ):
            if not np.array_equal(
                result.smoothed[:, column], result.preprocessed[:, column], equal_nan=True
            ):
                raise ValueError(
                    "A channel without respiratory smoothing differs from preprocessed"
                )
    if source_sha256(recording.source_path) != recording.source_sha256:
        raise ValueError("Source .dat checksum changed since import")
    with (path.parent / "validation.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "passed",
                "checked_at_utc": utc_now(),
                "schema_version": "dbat_manual_check_v3",
                "recording_id": recording.recording_id,
                "source_sha256": recording.source_sha256,
                "checks": [
                    "original samples/checksum unchanged",
                    "three canonical signals; no duplicate resampled bundle",
                    "explicit original and processed sampling rates/time axes",
                    "processed-grid Boolean masks",
                    "numerical availability independent of QC/support masks",
                    "smoothed respiratory source and processed rate",
                    "QC identity/bounds/provenance",
                    "exact HDF5/JSON round trip",
                ],
            },
            stream,
            indent=2,
        )
    logger.debug(
        "Round trip verified: %s | original=%s at %g Hz | processed=%s at %g Hz",
        recording.recording_id,
        result.original.shape,
        result.original_fs_hz,
        result.smoothed.shape,
        result.processed_fs_hz,
    )
    logger.debug("Validation: %s", path.parent / "validation.json")
    return summary


def check_recording(config, subject, measurement):
    """Load, prepare once, review, condition and verify one recording.

    Cancel returns None and leaves existing results and accepted QC untouched.
    All processing goes through the same DBAT package functions as a single run.
    """
    recording = load_dat_recording(
        config, config["input"]["dat_file"] or None, subject=subject, measurement=measurement
    )
    qc_file = config["qc"]["annotation_files"].get(recording.recording_id)
    qc = (
        load_qc_annotations(qc_file, recording)
        if qc_file
        else new_qc_annotations(recording, config["qc"]["reviewer"])
    )
    prepared = prepare_recording(recording, config)
    if config["qc"]["open_gui"]:
        from dbat.preprocessing.preproc_gui import run_qc_gui

        qc = run_qc_gui(recording, config, qc, prepared=prepared)
        if qc is None:
            return None
        save_qc_annotations(qc, recording, config["paths"]["output_dir"])

    result = preprocess_recording(recording, config, qc, prepared=prepared)
    path = save_preprocessing_result(result, config)

    verify_saved_result(path, result)
    return path


def main(config_file):
    """Run Phase 1 for the configured subject/measurement Cartesian product.

    Use project logging configured from run.verbosity.

    Parameters
    ----------
    config_file : path-like
        Grouped TOML containing selection, paths, QC and processing settings.

    Returns
    -------
    list of Path
        Completed HDF5 outputs. Excluded and GUI-cancelled recordings are skipped;
        input or processing errors stop the run with the original error.
    """
    config = load_config(config_file)
    selection = config["run"]
    configure_logging(selection["verbosity"])
    excluded = {tuple(pair) for pair in selection["exclude_recordings"]}
    outputs = []

    for subject in selection["subjects"]:
        for measurement in selection["measurements"]:
            if (subject, measurement) in excluded:
                logger.info(f"Skipping excluded Sub{subject}_M{measurement}")
                continue

            logger.info(f"Processing Sub{subject}_M{measurement}")
            output = check_recording(config, subject, measurement)
            if output is not None:
                outputs.append(output)

    logger.info(f"Completed {len(outputs)} recording(s).")
    return outputs


if __name__ == "__main__":
    # Edit this one path, then click Run/Debug without script parameters.
    config_file = REPOSITORY / "configs" / "default.toml"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=config_file, help="Grouped DBAT TOML configuration"
    )
    main(parser.parse_args().config)
