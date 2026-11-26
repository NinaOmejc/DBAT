"""Atomic landmark revisions adapted from local MAGMA-ml src/landmarks/io.py.

Source repository: https://github.com/NinaOmejc/magma-ml (local unpublished work).
Copyright (c) 2026 Nina Omejc. See this repository's MIT LICENSE.
"""

# Local reference src/landmarks/io.py SHA-256: 6bb22a3816b867d823d69a712b89da809d8ffbd8a9c6fd6c3c39256bddc322f1

import copy
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from .respiratory import validate_belt, waveform_hash

SCHEMA = "dbat_respiratory_landmarks_v1"


def _json_value(value):
    """Keep nonfinite scalar values explicit in strict JSON metadata."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": str(value)}
    return value


def _pack(value, handle):
    """Place arrays in native HDF5 datasets and return JSON-compatible metadata."""
    if isinstance(value, np.ndarray):
        name = f"arrays/{len(handle['arrays'])}"
        text = value.dtype.kind in "UO"
        if text and not all(isinstance(item, str) for item in value.flat):
            raise ValueError("Only numeric or text arrays may be serialized")
        options = {"compression": "gzip"} if value.ndim and value.size else {}
        handle.create_dataset(
            name,
            data=value.astype(object) if text else value,
            dtype=h5py.string_dtype() if text else None,
            **options,
        )
        return {"array_path": name}
    if isinstance(value, dict):
        packed = {}
        for key, item in value.items():
            if (
                key == "detection_waveform"
                and "source_waveform" in packed
                and np.array_equal(item, value["source_waveform"], equal_nan=True)
            ):
                packed[key] = packed["source_waveform"]
            else:
                packed[key] = _pack(item, handle)
        return packed
    if isinstance(value, (tuple, list)):
        return [_pack(item, handle) for item in value]
    return _json_value(value)


def _unpack(value, handle):
    """Restore native arrays and explicitly tagged nonfinite JSON values."""
    if isinstance(value, dict):
        if set(value) == {"array_path"}:
            dataset = handle[value["array_path"]]
            return dataset.asstr()[()] if h5py.check_string_dtype(dataset.dtype) else dataset[()]
        if set(value) == {"nonfinite_float"}:
            return float(value["nonfinite_float"])
        return {key: _unpack(item, handle) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item, handle) for item in value]
    return value


def save_landmarks(annotation, output_dir):
    """Atomically publish a new HDF5 revision; never replace any existing file.

    JSON metadata and compressed native datasets form one atomic artifact.
    The caller's committed state is untouched if serialization/publication fails.

    Parameters
    ----------
    annotation : dict
        A validated recording/belts landmark bundle.
    output_dir : path-like
        Parent directory for per-recording numbered revisions.

    Returns
    -------
    pathlib.Path
        New standalone HDF5 revision; source and previous files are unchanged.
    """
    if re.fullmatch(r"[A-Za-z0-9_-]+", annotation["identity"]["recording_id"]) is None:
        raise ValueError("Unsafe recording ID")
    for belt in annotation["belts"].values():
        validate_belt(belt, annotation["fs_hz"])
        if waveform_hash(belt["detection_waveform"]) != belt["waveform_hash"]:
            raise ValueError("Detection waveform changed; dependent landmarks are stale")
        if waveform_hash(belt["source_waveform"]) != belt["source_hash"]:
            raise ValueError("Immutable source waveform was modified")
    folder = Path(output_dir).resolve() / annotation["identity"]["recording_id"]
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{annotation['identity']['recording_id']}_landmarks_r"
    revisions = [
        int(path.stem[len(prefix) :])
        for path in folder.glob(f"{prefix}[0-9]*.h5")
        if path.stem[len(prefix) :].isdigit()
    ]
    revision = max([annotation.get("revision", 0), *revisions]) + 1
    destination = folder / f"{prefix}{revision:04d}.h5"
    saved = copy.deepcopy(annotation)
    saved.update(
        schema_version=SCHEMA,
        revision=revision,
        parent_revision=annotation.get("revision", 0),
        saved_at_utc=datetime.now(UTC).isoformat(),
    )
    fd, temporary = tempfile.mkstemp(prefix=".landmarks-", suffix=".h5", dir=folder)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                schema_version=SCHEMA,
                index_base=0,
                revision=revision,
                recording_id=saved["identity"]["recording_id"],
            )
            handle.create_group("arrays")
            metadata = _pack(saved, handle)
            handle.create_dataset(
                "metadata_json",
                data=json.dumps(metadata, allow_nan=False),
                dtype=h5py.string_dtype(),
            )
            handle.flush()
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)  # Atomic, and refuses an existing destination.
    finally:
        os.unlink(temporary)
    return destination


def load_landmarks(path):
    """Load and validate one DBAT landmark revision without rerunning detection."""
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA or handle.attrs.get("index_base") != 0:
            raise ValueError("Unsupported landmark schema or index convention")
        annotation = _unpack(json.loads(handle["metadata_json"].asstr()[()]), handle)
    if annotation["schema_version"] != SCHEMA:
        raise ValueError("Annotation metadata/schema mismatch")
    for belt in annotation["belts"].values():
        validate_belt(belt, annotation["fs_hz"])
        if (
            waveform_hash(belt["source_waveform"]) != belt["source_hash"]
            or waveform_hash(belt["detection_waveform"]) != belt["waveform_hash"]
        ):
            raise ValueError("Saved waveform hash mismatch")
        belt["source_waveform"].setflags(write=False)
        belt["detection_waveform"].setflags(write=False)
    return annotation
