"""Strict numeric ASCII import and immutable, versioned DBAT output files."""

import hashlib
import importlib.metadata
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from .. import __version__
from ..preprocessing.annotations import (
    recording_identity,
    utc_now,
    validate_qc_annotations,
)


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
    return Recording(
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


def _output_root(recording, output_dir):
    """Resolve an output root outside the actual source recording directory."""
    root = Path(output_dir).resolve()
    if root == recording.source_path.parent or root.is_relative_to(recording.source_path.parent):
        raise ValueError("Output must be outside the source recording directory")
    return root / recording.recording_id


def _version_name():
    """Create a portable UTC version name with a collision-resistant suffix."""
    return utc_now().replace(":", "").replace("+0000", "Z") + "_" + uuid4().hex[:8]


def save_qc_annotations(qc, recording, output_dir):
    """Commit accepted QC to a new JSON version before any conditioning runs."""
    validate_qc_annotations(qc, recording)
    if qc["accepted_at_utc"] is None:
        raise ValueError("Only explicitly accepted QC can be saved as a review")
    folder = _output_root(recording, output_dir) / "qc"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_version_name()}.json"
    temporary = path.with_suffix(".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(qc, stream, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.rename(path)
    return path


def load_qc_annotations(path, recording):
    """Reload a DBAT QC JSON or summary JSON after exact source/grid validation."""
    path = Path(path).resolve(strict=True)
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    qc = document["qc"] if document.get("schema_version") == "dbat_preprocessing_v1" else document
    validate_qc_annotations(qc, recording)
    qc["loaded_from_file"] = str(path)
    return qc


def result_metadata(result, config):
    """Build JSON-compatible provenance and per-channel native-grid QC summaries."""
    recording = result.recording
    return {
        "schema_version": config["output"]["schema_version"],
        "created_at_utc": utc_now(),
        "dbat_version": __version__,
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
                "artifact_interval_count": sum(
                    name in interval["channels"] for interval in result.qc["artifacts"]
                ),
                "artifact_duration_sec": int(result.masks["artifact"][:, column].sum())
                / recording.fs_hz,
                "reviewed_fraction": float(result.masks["reviewed"][:, column].mean()),
                "fully_reviewed": bool(result.masks["reviewed"][:, column].all()),
                "valid_fraction": float(result.masks["valid"][:, column].mean()),
                **result.provenance["counts"][name],
            }
            for column, name in enumerate(recording.channels)
        },
    }


def save_preprocessing_result(result, config):
    """Atomically publish a new run directory containing HDF5, JSON and preview.

    Existing runs and source files are never overwritten. Accepted QC is stored
    independently by the caller first, so a later processing failure cannot lose it.
    Returns the final HDF5 path. Incomplete writes remain in a .partial directory.
    """
    recording = result.recording
    validate_qc_annotations(result.qc, recording)
    root = _output_root(recording, config["paths"]["output_dir"]) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    version = _version_name()
    temporary = root / f".{version}.partial"
    temporary.mkdir()
    filename = config["output"]["filename"].format(recording_id=recording.recording_id)
    metadata = result_metadata(result, config)
    with h5py.File(temporary / filename, "x") as handle:
        handle.attrs["schema_version"] = config["output"]["schema_version"]
        handle.attrs["fs_hz"] = recording.fs_hz
        handle.attrs["array_axes"] = "samples,channels"
        handle.attrs["sample_intervals"] = "zero-based half-open [start,end)"
        handle.create_dataset("channels", data=list(recording.channels), dtype=h5py.string_dtype())
        for name, values in {
            "source/samples": recording.samples,
            "time_sec": recording.time_sec,
            "derived/conditioned": result.conditioned,
            "derived/respiratory_view": result.respiratory_view,
            **{f"masks/{name}": mask for name, mask in result.masks.items()},
        }.items():
            handle.create_dataset(name, data=values, compression="gzip", shuffle=True)
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
    return destination / filename


def save_preview(result, config, path):
    """Draw one configured viewport; display thinning never changes native arrays."""
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
    step = max(1, int(np.ceil((end - start) / config["qc"]["max_plot_points"])))
    indices = np.arange(start, end, step)
    figure = Figure(figsize=(13, 10), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(recording.channels), 1, sharex=True)
    for column, (axis, name) in enumerate(zip(axes, recording.channels, strict=True)):
        axis.plot(
            recording.time_sec[indices],
            recording.samples[indices, column],
            color="0.65",
            lw=0.7,
            label="Supplied",
        )
        axis.plot(
            recording.time_sec[indices],
            result.conditioned[indices, column],
            lw=0.7,
            label="Conditioned (unsmoothed)",
        )
        axis.plot(
            recording.time_sec[indices],
            result.respiratory_view[indices, column],
            lw=0.8,
            label="Respiratory view",
        )
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
    axes[-1].set_xlabel("Native time (seconds); supplied amplitude units preserved")
    figure.suptitle(
        f"{recording.recording_id} | {result.qc['review_status']} | Unknown QC is not clean; unsupported derived samples are NaN"
    )
    figure.savefig(path, dpi=130)
