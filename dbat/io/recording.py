"""Strict numeric ASCII import and immutable, versioned DBAT output files."""

import hashlib
import importlib.metadata
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np

from .. import __version__
from ..config import READABLE_RESULT_SCHEMAS, RESULT_SCHEMA
from ..logging import logger
from ..preprocessing.annotations import (
    annotation_masks,
    continuous_runs,
    new_qc_annotations,
    recording_identity,
    utc_now,
    validate_qc_annotations,
)


PREPROCESSING_ALGORITHM = "dbat_preprocessing_v7"


@dataclass(frozen=True)
class Recording:
    """Supplied float64 samples (N, C), native seconds and declared channel order."""

    samples: np.ndarray
    time_sec: np.ndarray
    channels: tuple[str, ...]
    fs_hz: float
    subject: int
    measurement: int
    source_path: Path
    source_sha256: str
    source_modified_at_utc: str

    @property
    def recording_id(self):
        """Return the stable subject/measurement identifier."""
        return f"Sub{self.subject}_M{self.measurement}"


def source_sha256(path):
    """Hash a source file read-only, without loading its bytes all at once."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_dat_recording(config, dat_path=None, *, subject=None, measurement=None):
    """Load one numeric .dat as (samples, configured channels), without reshaping.

    Parameters
    ----------
    config : dict
        Validated grouped settings declaring fs (Hz), column order and filenames.
    dat_path : path-like, optional
        Explicit .dat; otherwise resolve subject/measurement in paths.input_dir.
    subject, measurement : int, optional
        Positive identities, checked against SubN_PomM or SubN_MM when present.
        Required explicitly if the filename does not encode identity.

    Returns
    -------
    Recording
        Read-only loaded float64 values and arange(N)/fs timeline. Input files
        may already be externally detrended/normalized; no sensor-raw claim.

    Notes
    -----
    Import convention independently follows MATLAB numeric load/native time:
    https://github.com/NinaOmejc/magma-labeling/blob/main/src/utils/load_raw_data.m
    DBAT intentionally rejects flattened/transposed input instead of guessing.
    """
    for value, name in ((subject, "subject"), (measurement, "measurement")):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if dat_path is None:
        if subject is None or measurement is None:
            raise ValueError("Provide dat_path or both subject and measurement")
        candidates = [
            Path(config["paths"]["input_dir"])
            / config["input"][key].format(subject=subject, measurement=measurement)
            for key in ("filename_pattern", "alternate_filename_pattern")
            if config["input"][key]
        ]
        dat_path = next((path for path in candidates if path.is_file()), None)
        if dat_path is None:
            raise FileNotFoundError(
                f"No recording found; checked: {[str(path) for path in candidates]}"
            )
    path = Path(dat_path).resolve(strict=True)
    if path.suffix.lower() != ".dat" or not path.is_file():
        raise ValueError(f"Expected one existing numeric .dat file: {path}")
    match = re.search(r"(?:^|_)Sub(\d+)_(?:Pom|M)(\d+)(?:_|$)", path.stem, re.IGNORECASE)
    if match:
        parsed_subject, parsed_measurement = map(int, match.groups())
        if (subject is not None and subject != parsed_subject) or (
            measurement is not None and measurement != parsed_measurement
        ):
            raise ValueError("Explicit subject/measurement disagrees with the .dat filename")
        subject, measurement = parsed_subject, parsed_measurement
    if subject is None or measurement is None or subject <= 0 or measurement <= 0:
        raise ValueError("Filename lacks valid SubN_PomM identity; provide subject and measurement")
    fs_hz = float(config["input"]["fs_hz"])
    fs_match = re.search(r"(?:^|_)fs(\d+(?:\.\d+)?)(?:_|$)", path.stem, re.IGNORECASE)
    if fs_match and float(fs_match.group(1)) != fs_hz:
        raise ValueError(f"Filename sample rate disagrees with configured input.fs_hz={fs_hz}")
    logger.info("  Loading Sub%s_M%s...", subject, measurement)
    logger.debug("Source: %s", path)
    before = path.stat()
    try:
        samples = np.loadtxt(path, dtype=np.float64, comments=None, ndmin=2)
    except ValueError as error:
        raise ValueError(f"Malformed numeric ASCII .dat: {path}: {error}") from error
    channels = tuple(config["input"]["channels"])
    if samples.size == 0 or samples.shape[0] < 2 or samples.shape[1] != len(channels):
        raise ValueError(
            f"Expected at least two rows and {len(channels)} columns in {path}; got {samples.shape}. No transpose/reshape is inferred."
        )
    checksum = source_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while it was being read; retry with a stable file")
    from datetime import datetime, timezone

    times = np.arange(samples.shape[0], dtype=np.float64) / fs_hz
    samples.setflags(write=False)
    times.setflags(write=False)
    recording = Recording(
        samples,
        times,
        channels,
        fs_hz,
        subject,
        measurement,
        path,
        checksum,
        datetime.fromtimestamp(after.st_mtime, timezone.utc).isoformat(),
    )
    logger.debug(
        "Loaded %s | shape=%s | fs=%g Hz | duration=%g s",
        recording.recording_id, samples.shape, fs_hz, samples.shape[0] / fs_hz,
    )
    return recording


def _output_root(recording, output_dir):
    """Resolve an output root outside the actual source recording directory."""
    root = Path(output_dir).resolve()
    if root == recording.source_path.parent or root.is_relative_to(recording.source_path.parent):
        raise ValueError("Output must be outside the source recording directory")
    return root / recording.recording_id


def reserve_preprocessing_run(recording, output_dir):
    """Atomically reserve a new .runN.partial directory without reusing a number."""
    root = _output_root(recording, output_dir)
    root.mkdir(parents=True, exist_ok=True)
    number = 0
    for path in root.iterdir():
        match = re.fullmatch(r"\.?run(\d+)(?:\.partial)?", path.name)
        if match:
            number = max(number, int(match[1]) + 1)
    while True:
        pending = root / f".run{number}.partial"
        try:
            pending.mkdir()
        except FileExistsError:
            number += 1
            continue
        if not (root / f"run{number}").exists():
            return pending
        pending.rmdir()  # Only this call's empty reservation.
        number += 1


def _write_run_qc(qc, recording, output_dir, run_dir):
    """Write QC atomically in its reserved run, or require an exact existing copy."""
    folder = Path(run_dir).resolve(strict=True)
    if folder.parent != _output_root(recording, output_dir) or not re.fullmatch(
        r"\.run[0-9]+\.partial", folder.name
    ):
        raise ValueError("QC must be written in a reserved pending run for this recording")
    path = folder / "qc.json"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            if json.load(stream) != qc:
                raise ValueError("Pending run contains different accepted QC; reserve a new run")
        return path
    temporary = folder / "qc.json.partial"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(qc, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)
    return path


def save_qc_annotations(qc, recording, output_dir, *, run_dir=None):
    """Preserve accepted QC immediately in a recoverable pending run's qc.json.

    Returns the QC path; pass its parent to save_preprocessing_result(run_dir=...).
    Reused QC keeps its original decisions, reviewer and acceptance timestamp.
    """
    validate_qc_annotations(qc, recording)
    if qc["accepted_at_utc"] is None:
        raise ValueError("Only explicitly accepted QC can be saved as a review")
    logger.info("  Saving QC for %s...", recording.recording_id)
    folder = reserve_preprocessing_run(recording, output_dir) if run_dir is None else run_dir
    path = _write_run_qc(qc, recording, output_dir, folder)
    logger.debug("Accepted QC saved: %s", path)
    return path


def load_qc_annotations(path, recording):
    """Reload a DBAT QC JSON or summary JSON after exact source/grid validation."""
    path = Path(path).resolve(strict=True)
    logger.info("  Loading saved QC for %s...", recording.recording_id)
    logger.debug("QC source: %s", path)
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("QC JSON must contain an object")
    qc = document["qc"] if document.get("schema_version") in READABLE_RESULT_SCHEMAS else document
    if not isinstance(qc, dict):
        raise ValueError("QC annotations must contain an object")
    validate_qc_annotations(qc, recording)
    qc["loaded_from_file"] = str(path)
    return qc


def load_latest_qc_annotations(recording, config):
    """Find the newest compatible accepted review in runs, pending runs or legacy qc/.

    Explicit overrides fail strictly. Discovery validates native identity/hash/grid
    and orders accepted reviews by acceptance time, then saved-file time. Pending
    QC remains discoverable after interrupted processing; results are never moved.
    """
    override = config["qc"].get("annotation_files", {}).get(recording.recording_id)
    if override:
        return load_qc_annotations(override, recording)
    root = _output_root(recording, config["paths"]["output_dir"])
    paths = list((root / "qc").glob("*.json"))
    for folder in root.iterdir() if root.exists() else ():
        if folder.is_dir() and re.fullmatch(r"\.?run[0-9]+(?:\.partial)?", folder.name):
            paths.extend(folder.glob("qc.json"))
    latest, latest_key = None, None
    for path in paths:
        try:
            qc = load_qc_annotations(path, recording)
            if not qc["accepted_at_utc"]:
                continue
            accepted = datetime.fromisoformat(qc["accepted_at_utc"].replace("Z", "+00:00"))
            if accepted.tzinfo is None:
                raise ValueError("Acceptance timestamp must include its time zone")
            key = (accepted.timestamp(), path.stat().st_mtime_ns, str(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("  Skipping QC %s: %s", path, error)
            continue
        if latest_key is None or key > latest_key:
            latest, latest_key = qc, key
    return latest if latest is not None else new_qc_annotations(recording, config["qc"]["reviewer"])


def load_reusable_preprocessing_result(recording, config, qc):
    """Return (saved result dictionary, HDF5 path) for the newest compatible run.

    Only completed runN directories are considered, in descending numeric order.
    Source, grids, schema, algorithms, numerical settings and current QC must
    match; rejected runs are logged. Returns None when reuse is disabled or no
    compatible run exists. Existing files and saved provenance remain unchanged.
    """
    if not config["preprocessing"].get("reuse_preproc", True):
        return None
    from ..preprocessing.cleaning import ALGORITHM_VERSION

    validate_qc_annotations(qc, recording)
    settings = {
        key: value for key, value in config["preprocessing"].items() if key != "reuse_preproc"
    }
    ignored_qc = {"loaded_from_file"}
    if not qc["accepted_at_utc"]:
        ignored_qc.update(("created_at_utc", "reviewer"))
    expected_qc = {key: value for key, value in qc.items() if key not in ignored_qc}
    root = _output_root(recording, config["paths"]["output_dir"])
    folders = sorted(
        (
            path
            for path in root.glob("run*")
            if path.is_dir() and re.fullmatch(r"run[0-9]+", path.name)
        ),
        key=lambda path: int(path.name[3:]),
        reverse=True,
    )
    for folder in folders:
        try:
            paths = list(folder.glob("*.h5"))
            if len(paths) != 1 or not (folder / "summary.json").is_file():
                raise ValueError("run is incomplete or has ambiguous HDF5 results")
            saved = load_preprocessing_result(paths[0])
            metadata = saved["metadata"]
            if saved["schema_version"] == RESULT_SCHEMA:
                with (folder / "qc.json").open(encoding="utf-8") as stream:
                    if json.load(stream) != metadata["qc"]:
                        raise ValueError("Run QC and HDF5 metadata disagree")
            with (folder / "summary.json").open(encoding="utf-8") as stream:
                if json.load(stream) != metadata:
                    raise ValueError("HDF5 metadata and summary disagree")
            if (
                metadata["identity"] != recording_identity(recording)
                or metadata["source"]["sha256"] != recording.source_sha256
                or saved["original_fs_hz"] != recording.fs_hz
                or saved["channels"] != list(recording.channels)
                or recording.fs_hz != config["input"]["fs_hz"]
                or list(recording.channels) != config["input"]["channels"]
            ):
                raise ValueError(
                    "source identity/hash, native sampling rate or channel order changed"
                )
            previous = metadata["config"]
            if not isinstance(previous["preprocessing"], dict):
                raise ValueError("saved preprocessing settings must be an object")
            previous_settings = {
                key: value
                for key, value in previous["preprocessing"].items()
                if key != "reuse_preproc"
            }
            processing = metadata["processing"]
            if (
                previous_settings != settings
                or previous["qc"]["channels"] != config["qc"]["channels"]
                or previous["output"]["schema_version"] != config["output"]["schema_version"]
                or processing["algorithm_version"] != PREPROCESSING_ALGORITHM
                or processing["automatic_cleaning"]["algorithm_version"] != ALGORITHM_VERSION
            ):
                raise ValueError(
                    "preprocessing settings, QC channel selection or algorithms changed"
                )
            saved_qc = metadata["qc"]
            if not isinstance(saved_qc, dict):
                raise ValueError("saved QC annotations must be an object")
            validate_qc_annotations(saved_qc, recording)
            if {
                key: value for key, value in saved_qc.items() if key not in ignored_qc
            } != expected_qc:
                raise ValueError(
                    "saved QC differs from the latest accepted QC or explicit override"
                )
            decision = saved_qc.get("polarity")
            if processing["polarity"]["decision"] != decision or (
                decision
                and decision["preparation_signature"]
                != processing["automatic_cleaning"]["signature"]
            ):
                raise ValueError("saved polarity decision is incompatible with automatic cleaning")
            target_fs = settings["resampling"]["resample_freq"] or recording.fs_hz
            rate = processing["resampling"]
            count = len(recording.samples)
            if rate["applied"]:
                if any(type(rate[key]) is not int or rate[key] <= 0 for key in ("up", "down")):
                    raise ValueError("invalid saved resampling ratio")
                count = (count * rate["up"] + rate["down"] - 1) // rate["down"]
                if saved["processed_fs_hz"] != recording.fs_hz * rate["up"] / rate["down"]:
                    raise ValueError("saved sampling rate disagrees with resampling ratio")
            elif saved["processed_fs_hz"] != recording.fs_hz:
                raise ValueError("native result has a different processed sampling rate")
            if not np.isclose(saved["processed_fs_hz"], target_fs, rtol=1e-10, atol=0):
                raise ValueError("processed sampling rate changed")
            if (
                metadata["original_grid"]["fs_hz"] != recording.fs_hz
                or metadata["original_grid"]["n_samples"] != len(recording.samples)
                or metadata["processed_grid"]["fs_hz"] != saved["processed_fs_hz"]
                or metadata["processed_grid"]["n_samples"] != count
            ):
                raise ValueError("saved grid metadata disagrees with the source/processed grid")
            shape = (count, len(recording.channels))
            if (
                not np.array_equal(saved["original_time_sec"], recording.time_sec)
                or not np.array_equal(
                    saved["processed_time_sec"], np.arange(count) / saved["processed_fs_hz"]
                )
                or saved["preproc_signal"].shape != shape
            ):
                raise ValueError(
                    "saved signals or time grids do not match the source/processed grid"
                )
            required_masks = {
                "artifact",
                "reviewed",
                "interpolated",
                "auto_candidate",
                "unresolved_candidate",
                "finite",
                "quality_selected",
                "uncertain",
                "eligible",
                "preprocessed_support",
                "preprocessed_interpolated",
                "preprocessed_valid",
                "smoothed_support",
                "smoothed_interpolated",
                "valid",
            }
            if rate["applied"]:
                required_masks.update(("boundary", "missing_influence"))
            if not required_masks <= saved["masks"].keys() or any(
                mask.shape != shape or mask.dtype != bool for mask in saved["masks"].values()
            ):
                raise ValueError("saved QC/support masks are missing or use a different grid")
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("  Not reusing %s: %s", folder, error)
            continue
        logger.info("  Reusing preprocessing for %s: %s", recording.recording_id, paths[0])
        return saved, paths[0]
    return None


def load_preprocessing_result(path):
    """Read the final signal and metadata from v4 or earlier DBAT HDF5 schemas.

    v4 contains preproc_signal only. Earlier stored arrays remain available in
    signals (and resampled where present), without changing or migrating files.
    preproc_signal always identifies the final waveform on the returned processed
    grid; metadata and all original saved arrays/masks are preserved.
    """
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a DBAT HDF5 file, not a directory: {path}")
    with h5py.File(path, "r") as handle:
        schema = handle.attrs["schema_version"]
        if schema not in READABLE_RESULT_SCHEMAS:
            raise ValueError(f"Unsupported DBAT result schema: {schema}")
        metadata = json.loads(handle["metadata_json"].asstr()[()])
        if metadata["schema_version"] != schema:
            raise ValueError("HDF5 and embedded metadata schemas disagree")
        result = {
            "schema_version": schema,
            "metadata": metadata,
            "channels": handle["channels"].asstr()[:].tolist(),
            "masks": {name: dataset[:] for name, dataset in handle["masks"].items()},
        }
        if schema in {"dbat_preprocessing_v3", RESULT_SCHEMA}:
            result.update(
                original_time_sec=handle["original_time_sec"][:],
                processed_time_sec=handle["processed_time_sec"][:],
                original_fs_hz=float(handle.attrs["original_fs_hz"]),
                processed_fs_hz=float(handle.attrs["processed_fs_hz"]),
            )
        else:
            result.update(
                original_time_sec=handle["time_sec"][:],
                processed_time_sec=handle["time_sec"][:],
                original_fs_hz=float(handle.attrs["fs_hz"]),
                processed_fs_hz=float(handle.attrs["fs_hz"]),
            )
        if schema == RESULT_SCHEMA:
            result["preproc_signal"] = handle["preproc_signal"][:]
            result["respiratory_analysis_source"] = "preproc_signal"
        else:
            if schema == "dbat_preprocessing_v1":
                signals = {
                    "original": handle["source/samples"][:],
                    "preprocessed": handle["derived/conditioned"][:],
                    "smoothed": handle["derived/respiratory_view"][:],
                }
                if "derived/corrected" in handle:
                    signals["corrected"] = handle["derived/corrected"][:]
            else:
                signals = {name: dataset[:] for name, dataset in handle["signals"].items()}
            signals["original"].setflags(write=False)
            result["signals"] = signals
            result["preproc_signal"] = signals["smoothed"]
            result["respiratory_analysis_source"] = "signals/smoothed"
            if "resampled" in handle:
                group = handle["resampled"]
                names = (
                    {"preprocessed": "conditioned", "smoothed": "respiratory_view"}
                    if schema == "dbat_preprocessing_v1"
                    else {"preprocessed": "signals/preprocessed", "smoothed": "signals/smoothed"}
                )
                result["resampled"] = {
                    "signals": {name: group[key][:] for name, key in names.items()},
                    "time_sec": group["time_sec"][:],
                    "fs_hz": float(group.attrs["fs_hz"]),
                    "masks": {name: dataset[:] for name, dataset in group["masks"].items()},
                }
    return result


def result_metadata(result, config):
    """Describe both grids; manual coverage is native and validity is processed."""
    recording = result.recording
    native_qc = annotation_masks(result.qc, recording)
    return {
        "schema_version": config["output"]["schema_version"],
        "created_at_utc": utc_now(),
        "dbat_version": __version__,
        "signal_definitions": {
            "preproc_signal": "Final corrected, polarity-adjusted, optionally detrended/filtered/resampled and once-smoothed signal",
        },
        "respiratory_analysis_source": "preproc_signal",
        "original_grid": {"fs_hz": result.original_fs_hz, "time_axis": "original_time_sec",
                          "n_samples": len(result.original), "mask_basis": "Native QC intervals in qc"},
        "processed_grid": {"fs_hz": result.processed_fs_hz, "time_axis": "processed_time_sec",
                           "n_samples": len(result.preproc_signal), "mask_basis": "All masks/*"},
        "channel_summary_basis": "Manual coverage/artifact durations use native QC; mask counts/valid fractions use the processed grid",
        "import_validation": "passed: numeric matrix, identity, channels and configured sampling rate",
        "identity": recording_identity(recording),
        "source": {
            "path": str(recording.source_path),
            "sha256": recording.source_sha256,
            "modified_at_utc": recording.source_modified_at_utc,
            "description": "Supplied MAGMA input; may already be detrended/normalized",
            "sampling_rate_basis": "Declared configuration, checked against filename fs token when present",
            "channel_mapping_basis": "Explicit configured ASCII column order; no header inference",
            "column_mapping_zero_based": dict(
                zip(recording.channels, range(len(recording.channels)), strict=True)
            ),
        },
        "duration_sec": recording.samples.shape[0] / recording.fs_hz,
        "last_sample_sec": float(recording.time_sec[-1]),
        "config": config,
        "qc": result.qc,
        "processing": result.provenance,
        "runtime": {
            "python": sys.version,
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "matplotlib", "h5py")
            },
        },
        "channels": {
            name: {
                "rating": result.qc["channel_ratings"][name],
                "qc_selected": name in config["qc"]["channels"],
                "artifact_interval_count": sum(
                    name in interval["channels"] for interval in result.qc["artifacts"]
                ),
                "artifact_duration_sec": int(native_qc["artifact"][:, column].sum())
                / recording.fs_hz,
                "reviewed_fraction": float(native_qc["reviewed"][:, column].mean()),
                "fully_reviewed": bool(native_qc["reviewed"][:, column].all()),
                "valid_fraction": float(result.masks["valid"][:, column].mean()),
                **result.provenance["counts"][name],
            }
            for column, name in enumerate(recording.channels)
        },
    }


def save_preprocessing_result(result, config, *, run_dir=None):
    """Publish a new immutable run with final HDF5, exact qc.json and summary.

    Pass the pending directory from save_qc_annotations to retain accepted QC on
    processing/write failure. Direct callers reserve a run here; unreviewed QC is
    stored as explicitly unaccepted. Failed writes remain recoverable in place.
    """
    if result.provenance.get("provisional"):
        raise ValueError("A provisional GUI preview cannot be saved; recompute accepted QC first")
    if config["output"]["schema_version"] != RESULT_SCHEMA:
        raise ValueError(f"New results must use {RESULT_SCHEMA}")
    recording = result.recording
    validate_qc_annotations(result.qc, recording)
    logger.info("  Saving preprocessing result for %s...", recording.recording_id)
    root = _output_root(recording, config["paths"]["output_dir"])
    temporary = reserve_preprocessing_run(recording, config["paths"]["output_dir"]) if run_dir is None else Path(run_dir)
    _write_run_qc(result.qc, recording, config["paths"]["output_dir"], temporary)
    version = temporary.name.removeprefix(".").removesuffix(".partial")
    filename = config["output"]["filename"].format(recording_id=recording.recording_id)
    metadata = result_metadata(result, config)
    with h5py.File(temporary / filename, "x") as handle:
        handle.attrs["schema_version"] = config["output"]["schema_version"]
        handle.attrs["original_fs_hz"] = result.original_fs_hz
        handle.attrs["processed_fs_hz"] = result.processed_fs_hz
        handle.attrs["masks_grid"] = "processed_time_sec"
        handle.attrs["qc_intervals_grid"] = "original_time_sec"
        handle.attrs["array_axes"] = "samples,channels"
        handle.attrs["sample_intervals"] = "zero-based half-open [start,end)"
        handle.create_dataset("channels", data=list(recording.channels), dtype=h5py.string_dtype())
        for name, values in {
            "preproc_signal": result.preproc_signal,
            "original_time_sec": result.original_time_sec,
            "processed_time_sec": result.processed_time_sec,
            **{f"masks/{name}": mask for name, mask in result.masks.items()},
        }.items():
            handle.create_dataset(name, data=values, compression="gzip", shuffle=True)
        handle["preproc_signal"].attrs["fs_hz"] = result.processed_fs_hz
        handle["preproc_signal"].attrs["time_axis"] = "processed_time_sec"
        handle.create_dataset(
            "metadata_json",
            data=json.dumps(metadata, ensure_ascii=False, allow_nan=False),
            dtype=h5py.string_dtype(),
        )
    with (temporary / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
    if config["output"]["save_preview"]:
        save_preview(result, config, temporary / "preview.png")
    destination = root / version
    temporary.rename(destination)
    logger.debug(
        "QC: %s | %s original artifact intervals (union durations below)",
        result.qc["review_status"], len(result.qc["artifacts"]),
    )
    for name, info in metadata["channels"].items():
        if not info["qc_selected"]:
            continue
        logger.debug(
            "  %s: %s; reviewed=%.1f%%; artifacts=%s / %g s; valid=%s/%s (%.1f%%); supported=%s",
            name, info["rating"], 100 * info["reviewed_fraction"],
            info["artifact_interval_count"], info["artifact_duration_sec"],
            info["valid"], len(result.smoothed), 100 * info["valid_fraction"],
            info["smoothed_support"],
        )
    logger.debug("Output: %s", destination / filename)
    logger.debug("Metadata: %s", destination / "summary.json")
    return destination / filename


def save_preview(result, config, path):
    """Preview selected QC channels; drawing never changes stored arrays."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    recording = result.recording
    start = int(config["output"]["preview_start_sec"] * recording.fs_hz)
    if start >= recording.samples.shape[0]:
        raise ValueError("output.preview_start_sec lies beyond this recording")
    end = min(
        recording.samples.shape[0],
        start + max(2, int(config["output"]["preview_duration_sec"] * recording.fs_hz)),
    )
    first, last = np.searchsorted(result.processed_time_sec, [start / recording.fs_hz, end / recording.fs_hz])
    stride = max(1, int(np.ceil((last - first) / config["qc"]["max_plot_points"])))
    processed_indices = np.arange(first, last, stride)
    figure = Figure(figsize=(13, 10), layout="constrained")
    FigureCanvasAgg(figure)
    channels = config["qc"]["channels"]
    axes = figure.subplots(len(channels), 1, sharex=True, squeeze=False)[:, 0]
    for axis, name in zip(axes, channels, strict=True):
        column = recording.channels.index(name)
        axis.plot(
            result.processed_time_sec[processed_indices], result.preproc_signal[processed_indices, column],
            lw=0.8, label="Final preprocessed signal",
        )
        for key, color in (("interpolated", "teal"), ("unresolved_candidate", "orange")):
            for begin, finish in continuous_runs(result.masks[key][first:last, column]):
                axis.axvspan((first + begin) / result.processed_fs_hz, (first + finish) / result.processed_fs_hz,
                             color=color, alpha=0.18)
        for interval in result.qc["artifacts"]:
            if (
                name in interval["channels"]
                and interval["end_sample"] > start
                and interval["start_sample"] < end
            ):
                axis.axvspan(
                    interval["start_sec"], interval["end_sec"], color="tab:red", alpha=0.25
                )
        axis.set_ylabel(f"{name}\n{result.qc['channel_ratings'][name]}", fontsize=8)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlim(start / recording.fs_hz, end / recording.fs_hz)
    axes[-1].set_xlabel("Time (seconds); supplied amplitude units preserved")
    figure.suptitle(
        f"{recording.recording_id} | native {result.original_fs_hz:g} Hz; processed {result.processed_fs_hz:g} Hz | {result.qc['review_status']} | Shading: teal = interpolated; orange = candidate"
    )
    figure.savefig(path, dpi=130)


def preprocessing_provenance(result, config, prepared, decision):
    """Assemble the unchanged preprocessing provenance for saved results.

    Stage diagnostics arrive in result.provenance. Returns the original
    complete dictionary, including processing order, grids, QC and mask counts,
    and emits the existing final debug messages. No signal values are changed.
    """
    recording = result.recording
    masks = result.masks
    settings = config["preprocessing"]
    diagnostics = result.provenance
    processors = diagnostics["channel_processors"]
    detrend_parameters = diagnostics["detrend"]
    filter_details = diagnostics["filtering"]
    smoothing_parameters = diagnostics["smoothing"]
    resampling = diagnostics["resampling"]
    processing_skips = diagnostics["processing_skips"]
    processed_fs, times = result.processed_fs_hz, result.processed_time_sec
    choice = decision["choice"] if decision else None
    provenance = {
        "algorithm_version": PREPROCESSING_ALGORITHM,
        "enabled": settings["enabled"],
        "selected_channels": settings["channels"],
        "qc_channels": config["qc"]["channels"],
        "channel_processors": processors,
        "detrend": detrend_parameters,
        "filtering": filter_details,
        "smoothing": smoothing_parameters,
        "resampling": resampling,
        "processing_order": "automatic correction; source-relative polarity; native detrend; native highpass; native lowpass; optional resampling; one processed-grid smoothing pass",
        "edge_policy": "Moving mean shrinks at recording ends with valid numerical support; genuine gap/artifact influence stays unsupported. Optional IIR filters cross QC/artifact flags; reflection guards at recording ends or genuine missing gaps remain unsupported.",
        "signal_policy": "Numerical conditioning is independent of channel ratings, QC channel selection and reviewed coverage",
        "processing_skips": processing_skips,
        "qc_preview_policy": "When needed, QC reviews complete provisional preprocessing; accepted decisions are recomputed before saving",
        "respiratory_analysis_source": "preproc_signal",
        "respiratory_analysis_fs_hz": processed_fs,
        "respiratory_analysis_policy": "Every future respiratory detector/analysis uses preproc_signal, processed_fs_hz and valid/interpolated masks; no second general smoothing pass",
        "native_landmark_indices_compatible": processed_fs == recording.fs_hz
        and np.array_equal(times, recording.time_sec),
        "native_landmark_policy": "Require exact native-grid equality before direct historical MATLAB index reuse; otherwise reject direct reuse",
        "matlab_difference": "All respiratory reference and spectral inputs use smoothed; MATLAB also uses unsmoothed recordings for some calculations",
        "quality_policy": "Only selected QC channels with explicitly inspected support and usable ratings are eligible",
        "automatic_cleaning": {"signature": prepared["signature"], **prepared["provenance"]},
        "polarity": {
            "assessment": prepared["polarity"],
            "decision": decision,
            "applied_to": choice,
        },
        "limitations": [
            "No MATLAB numerical comparison performed",
            "Review coverage may be partial",
            "Automatic candidates are heuristic; interpolation is estimated, not newly inspected evidence",
        ],
        "counts": {
            name: {key: int(mask[:, column].sum()) for key, mask in masks.items()}
            for column, name in enumerate(recording.channels)
        },
    }
    logger.debug("Smoothing: %s | Detrend: %s", smoothing_parameters, detrend_parameters)
    logger.debug("Resampling: %s | Filtering: %s", resampling, filter_details)
    return provenance
