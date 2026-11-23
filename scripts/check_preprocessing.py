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

from dbat.config import load_config
from dbat.io.recording import (
    load_dat_recording,
    load_qc_annotations,
    save_preprocessing_result,
    save_qc_annotations,
    source_sha256,
)
from dbat.preprocessing.annotations import (
    new_qc_annotations,
    utc_now,
    validate_qc_annotations,
)
from dbat.preprocessing.pipeline import preprocess_recording

REPOSITORY = Path(__file__).resolve().parents[1]


def verify_saved_result(path, result):
    """Reopen the real output; raise on lost samples, time, masks or metadata."""
    recording = result.recording
    with (path.parent / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    with h5py.File(path, "r") as handle:
        metadata = json.loads(handle["metadata_json"].asstr()[()])
        if metadata != summary or metadata["schema_version"] != "dbat_preprocessing_v1":
            raise ValueError("HDF5 and JSON metadata/schema disagree")
        if handle.attrs["fs_hz"] != recording.fs_hz or handle["channels"].asstr()[
            :
        ].tolist() != list(recording.channels):
            raise ValueError("Sampling rate or channel order changed during saving")
        expected_arrays = {
            "source/samples": recording.samples,
            "time_sec": recording.time_sec,
            "derived/conditioned": result.conditioned,
            "derived/respiratory_view": result.respiratory_view,
            **{f"masks/{name}": mask for name, mask in result.masks.items()},
        }
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
                "schema_version": "dbat_manual_check_v1",
                "recording_id": recording.recording_id,
                "source_sha256": recording.source_sha256,
                "checks": [
                    "source samples unchanged",
                    "native time grid",
                    "channel order/fs",
                    "derived arrays and boolean masks",
                    "QC identity/bounds/provenance",
                    "HDF5/JSON agreement",
                ],
            },
            stream,
            indent=2,
        )
    return summary


def check_recording(config, subject, measurement):
    """Load, review, condition and verify one recording; return its HDF5 path.

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
    if config["qc"]["open_gui"]:
        from dbat.preprocessing.qc_gui import run_qc_gui

        qc = run_qc_gui(recording, config, qc)
        if qc is None:
            print(
                f"Cancelled {recording.recording_id}: no new files saved; continuing to the next recording."
            )
            return None
        qc_path = save_qc_annotations(qc, recording, config["paths"]["output_dir"])
        print(f"Accepted QC saved before conditioning: {qc_path}")
    result = preprocess_recording(recording, config, qc)
    path = save_preprocessing_result(result, config)
    summary = verify_saved_result(path, result)
    print(
        f"Round trip verified: {recording.recording_id} | shape={recording.samples.shape} | "
        f"fs={recording.fs_hz:g} Hz | duration={summary['duration_sec']:g} s"
    )
    print(
        f"QC: {qc['review_status']} | {len(qc['artifacts'])} original artifact intervals (union durations below)"
    )
    for name, info in summary["channels"].items():
        print(
            f"  {name}: {info['rating']}; reviewed={info['reviewed_fraction']:.1%}; "
            f"artifacts={info['artifact_interval_count']} / {info['artifact_duration_sec']:g} s; "
            f"valid={info['valid']}/{recording.samples.shape[0]} ({info['valid_fraction']:.1%}); "
            f"supported={info['respiratory_support']}"
        )
    print(f"Smoothing: {result.provenance['smoothing']} | Detrend: {result.provenance['detrend']}")
    if qc["review_status"] == "not_reviewed":
        print(
            "No inspected QC evidence: selected conditioning is deferred; valid samples remain zero."
        )
    print(
        f"Output: {path}\nMetadata/QA: {path.parent / 'summary.json'} | {path.parent / 'validation.json'}"
    )
    return path


def main(config_file):
    """Run Phase 1 for the configured subject/measurement Cartesian product.

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
    excluded = {tuple(pair) for pair in selection["exclude_recordings"]}
    outputs = []
    # Same selection convention as MATLAB run_magma.m: all combinations, not zip.
    # https://github.com/NinaOmejc/magma-labeling/blob/main/src/run_magma.m
    for subject in selection["subjects"]:
        for measurement in selection["measurements"]:
            if (subject, measurement) in excluded:
                print(f"Skipping excluded Sub{subject}_M{measurement}")
                continue
            print(f"Processing Sub{subject}_M{measurement}")
            output = check_recording(config, subject, measurement)
            if output is not None:
                outputs.append(output)
    print(f"Completed {len(outputs)} recording(s).")
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
