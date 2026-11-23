"""Manually inspect ONE real .dat through DBAT's actual Phase-1 package path.

Run from any directory, e.g.:
    python scripts/check_preprocessing.py --subject 33 --measurement 1 --gui
Writes a versioned HDF5/JSON result and optional preview; validates its round trip.
No synthetic recordings, automated test suite or separate preprocessing pipeline.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
# Load the flat source directory under its public package name without installation.
package_spec = importlib.util.spec_from_file_location("dbat", REPOSITORY / "src" / "__init__.py")
package = importlib.util.module_from_spec(package_spec)
sys.modules["dbat"] = package
package_spec.loader.exec_module(package)

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


def main():
    """Load, optionally review, condition, save and reopen one supplied recording."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dat", type=Path, help="Explicit numeric .dat; identity checked against its filename"
    )
    parser.add_argument("--subject", type=int)
    parser.add_argument("--measurement", type=int)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs" / "default.toml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--qc", type=Path, help="Accepted DBAT QC JSON or previous run's summary.json"
    )
    parser.add_argument("--reviewer", help="Reviewer identity used by the manual GUI")
    gui = parser.add_mutually_exclusive_group()
    gui.add_argument("--gui", dest="gui", action="store_true")
    gui.add_argument("--no-gui", dest="gui", action="store_false")
    parser.set_defaults(gui=None)
    parser.add_argument(
        "--no-preview", action="store_true", help="Skip the optional static inspection image"
    )
    args = parser.parse_args()
    overrides = {"paths": {}, "qc": {}, "output": {}}
    if args.output_dir is not None:
        overrides["paths"]["output_dir"] = str(args.output_dir.resolve())
    if args.gui is not None:
        overrides["qc"]["open_gui"] = args.gui
    if args.reviewer is not None:
        overrides["qc"]["reviewer"] = args.reviewer
    if args.no_preview:
        overrides["output"]["save_preview"] = False
    config = load_config(args.config, overrides, explicit_dat=args.dat is not None)
    recording = load_dat_recording(
        config, args.dat, subject=args.subject, measurement=args.measurement
    )
    qc = (
        load_qc_annotations(args.qc, recording)
        if args.qc
        else new_qc_annotations(recording, config["qc"]["reviewer"])
    )
    if config["qc"]["open_gui"]:
        from dbat.preprocessing.qc_gui import run_qc_gui

        qc = run_qc_gui(recording, config, qc)
        if qc is None:
            print(
                "Cancelled: no new QC or preprocessing result saved; existing accepted reviews unchanged."
            )
            return
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


if __name__ == "__main__":
    main()
