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
    continuous_runs,
    new_qc_annotations,
    utc_now,
    validate_qc_annotations,
)
from dbat.preprocessing.pipeline import prepare_recording, preprocess_recording

REPOSITORY = Path(__file__).resolve().parents[1]


def verify_saved_result(path, result):
    """Reopen the real output; raise on lost samples, time, masks or metadata."""
    recording = result.recording
    logger.info("  Verifying saved result for %s...", recording.recording_id)
    with (path.parent / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    with h5py.File(path, "r") as handle:
        metadata = json.loads(handle["metadata_json"].asstr()[()])
        if metadata != summary or metadata["schema_version"] != RESULT_SCHEMA or handle.attrs["schema_version"] != RESULT_SCHEMA:
            raise ValueError("HDF5 and JSON metadata/schema disagree")
        if handle.attrs["fs_hz"] != recording.fs_hz or handle["channels"].asstr()[
            :
        ].tolist() != list(recording.channels):
            raise ValueError("Sampling rate or channel order changed during saving")
        if set(handle["signals"]) != {"original", "preprocessed", "smoothed"} or "derived" in handle or "source" in handle:
            raise ValueError("New results must contain exactly three canonical signal representations")
        expected_arrays = {
            "signals/original": result.original,
            "signals/preprocessed": result.preprocessed,
            "signals/smoothed": result.smoothed,
            "time_sec": recording.time_sec,
            **{f"masks/{name}": mask for name, mask in result.masks.items()},
        }
        source = recording.samples
        choice = result.provenance["polarity"]["applied_to"]
        oriented_source = source.copy()
        if choice is not None:
            oriented_source[:, recording.channels.index(choice)] *= -1
        observed = ~result.masks["interpolated"].copy()
        detrend = result.provenance["detrend"]
        if detrend["method"] != "none":
            for column, name in enumerate(recording.channels):
                if result.provenance["channel_processors"][name] != "belt_conditioning":
                    continue
                for start, end in continuous_runs(result.masks["eligible"][:, column]):
                    if end - start >= detrend["window_samples"]:
                        observed[start:end, column] = False
        if not np.array_equal(result.preprocessed[observed], oriented_source[observed], equal_nan=True):
            raise ValueError("Preserved values disagree with source-relative polarity outside interpolation/detrending")
        source_finite = np.isfinite(source)
        for values in (result.preprocessed, result.smoothed):
            if values.shape != source.shape or np.any(source_finite & ~np.isfinite(values)):
                raise ValueError("Processing changed the native shape or introduced nonfinite values at observed samples")
            if not np.array_equal(values[~source_finite], oriented_source[~source_finite], equal_nan=True):
                raise ValueError("Processing changed genuine nonfinite source values")
        if result.respiratory_signal is not result.smoothed:
            raise ValueError("Respiratory analysis must resolve directly to the single smoothed waveform")
        for column, name in enumerate(recording.channels):
            if name not in BELTS or result.provenance["smoothing"]["method"] == "none":
                if not np.array_equal(result.smoothed[:, column], result.preprocessed[:, column], equal_nan=True):
                    raise ValueError("A channel without respiratory smoothing differs from preprocessed")
        if np.any(result.masks["valid"] & (~result.masks["reviewed"] | result.masks["artifact"])):
            raise ValueError("Processing fabricated reviewed/clean analysis support")
        if result.resampled is not None:
            resampled = result.resampled
            if handle["resampled"].attrs["fs_hz"] != resampled["fs_hz"]:
                raise ValueError("Resampled sampling rate changed during saving")
            expected_arrays.update({
                f"resampled/signals/{name}": resampled[name]
                for name in ("preprocessed", "smoothed")
            })
            expected_arrays["resampled/time_sec"] = resampled["time_sec"]
            expected_arrays.update({
                f"resampled/masks/{name}": mask for name, mask in resampled["masks"].items()
            })
            n_samples = resampled["preprocessed"].shape[0]
            if (
                resampled["preprocessed"].shape != (n_samples, len(recording.channels))
                or resampled["smoothed"].shape != resampled["preprocessed"].shape
                or not np.array_equal(resampled["time_sec"], np.arange(n_samples) / resampled["fs_hz"])
            ):
                raise ValueError("Resampled channels do not share the declared zero-based time grid")
            rate = result.provenance["resampling"]
            numerator = np.arange(n_samples) * rate["down"]
            left = np.minimum(numerator // rate["up"], source.shape[0] - 1)
            right = np.minimum((numerator + rate["up"] - 1) // rate["up"], source.shape[0] - 1)
            available = source_finite[left] & source_finite[right]
            for name in ("preprocessed", "smoothed"):
                if np.any(available & ~np.isfinite(resampled[name])):
                    raise ValueError("Resampling introduced nonfinite values without genuine missing source support")
                if np.any(~available & np.isfinite(resampled[name])):
                    raise ValueError("Resampling fabricated numerical measurements inside a genuine source gap")
            masks = resampled["masks"]
            if np.any(masks["valid"] & (~masks["reviewed"] | masks["artifact"] | masks["boundary"] | masks["missing_influence"])):
                raise ValueError("Resampling accepted contaminated or unreviewed FIR support")
        elif "resampled" in handle:
            raise ValueError("Unexpected resampled data when resampling was disabled")
        for name, expected in expected_arrays.items():
            saved = handle[name][:]
            if (
                saved.dtype != expected.dtype
                or saved.shape != expected.shape
                or not np.array_equal(saved, expected, equal_nan=True)
            ):
                raise ValueError(f"Round-trip data mismatch: {name}")
        if not np.array_equal(
            handle["time_sec"][:], np.arange(recording.samples.shape[0]) / recording.fs_hz
        ):
            raise ValueError("Saved time does not match the native zero-based sample grid")
        if metadata["qc"] != result.qc:
            raise ValueError("Saved QC annotations differ from the accepted input")
        validate_qc_annotations(metadata["qc"], recording)
    if source_sha256(recording.source_path) != recording.source_sha256:
        raise ValueError("Source .dat checksum changed since import")
    with (path.parent / "validation.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "passed",
                "checked_at_utc": utc_now(),
                "schema_version": "dbat_manual_check_v2",
                "recording_id": recording.recording_id,
                "source_sha256": recording.source_sha256,
                "checks": [
                    "original source samples unchanged",
                    "canonical original/preprocessed/smoothed schema",
                    "numerical availability independent of QC/support masks",
                    "one authoritative smoothed respiratory input",
                    "native time grid",
                    "channel order/fs",
                    "native and optional resampled derived arrays, boolean masks and grids",
                    "automatic correction masks, source-relative polarity and independent manual coverage",
                    "QC identity/bounds/provenance",
                    "HDF5/JSON agreement",
                ],
            },
            stream,
            indent=2,
        )
    logger.debug(
        "Round trip verified: %s | shape=%s | fs=%g Hz | duration=%g s",
        recording.recording_id, recording.samples.shape, recording.fs_hz, summary["duration_sec"],
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
        from dbat.preprocessing.qc_gui import run_qc_gui

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
