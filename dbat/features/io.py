"""Exact saved-input validation and immutable per-recording feature persistence."""

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from ..config import BELTS
from ..io.recording import load_preprocessing_result, source_sha256
from ..logging import logger
from ..peak_detection.io import load_landmarks
from .reference import SCHEMA as REFERENCE_SCHEMA

SCHEMA = "dbat_physiological_features_v1"


def _same(a, b):
    """Compare loaded scientific bundles without ambiguous NumPy equality."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return (
            a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
            if a.dtype.kind not in "UOS"
            else np.array_equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def load_inputs(data, landmarks, references):
    """Load paths or validate loaded dictionaries against exact saved input revisions.

    Plain in-memory references are fingerprinted using Stage 5a's exact JSON
    serialization. Their recorded preprocessing/landmark paths must still exist;
    explicit file paths support relocated inputs. No GUI or calculation runs.
    """
    if isinstance(references, dict):
        ref = references
        encoded = json.dumps(ref, indent=2, allow_nan=False).encode("utf-8")
        reference_file = {
            "path": None,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "hash_basis": "Stage 5a JSON serialization",
        }
    else:
        path = Path(references).resolve(strict=True)
        encoded = path.read_bytes()
        ref = json.loads(encoded)
        reference_file = {
            "path": str(path),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "hash_basis": "file_bytes",
        }
    if ref["schema_version"] != REFERENCE_SCHEMA:
        raise ValueError("Stage 5b requires dbat_static_physiological_reference_v1")
    provenance = {"references": reference_file}
    bundles = []
    for name, supplied, loader in (
        ("preprocessing", data, load_preprocessing_result),
        ("landmarks", landmarks, load_landmarks),
    ):
        source = ref["provenance"][name]
        path = Path(source["path"] if isinstance(supplied, dict) else supplied).resolve(
            strict=True
        )
        digest = source_sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"Stale/incompatible references: {name} SHA-256 mismatch")
        loaded = loader(path)
        if isinstance(supplied, dict) and not _same(supplied, loaded):
            raise ValueError(f"In-memory {name} differs from its exact saved revision")
        if source_sha256(path) != digest:
            raise ValueError(f"{name} changed during loading")
        provenance[name] = {
            "path": str(path),
            "sha256": digest,
            "schema_version": loaded["schema_version"],
        }
        if source["schema_version"] != loaded["schema_version"]:
            raise ValueError(f"Reference {name} schema disagrees with its saved input")
        bundles.append(loaded)
    data, lm = bundles
    identity = data["metadata"]["identity"]
    if (
        identity != lm["identity"]
        or identity != ref["identity"]
        or lm["preprocessing"].get("result_sha256")
        != provenance["preprocessing"]["sha256"]
        or lm["revision"] != ref["provenance"]["landmarks"]["revision"]
        or lm["review_status"] != "manual_reviewed"
        or lm["source"] != "preproc_signal"
        or lm["preprocessing"]["qc"] != data["metadata"]["qc"]
    ):
        raise ValueError(
            "Preprocessing, reviewed landmarks and references do not describe the same revisions"
        )
    fs, time = data["processed_fs_hz"], data["processed_time_sec"]
    values, masks = data["preproc_signal"], data["masks"]
    if (
        fs != lm["fs_hz"]
        or fs != ref["processed_fs_hz"]
        or len(time) != ref["n_processed_samples"]
        or not len(time)
        or values.shape != (len(time), len(data["channels"]))
        or not np.array_equal(time, lm["time_sec"])
        or not np.allclose(time, np.arange(len(time)) / fs, rtol=0, atol=1e-9)
    ):
        raise ValueError("Input processed grids disagree")
    for name in ("valid", "artifact", "uncertain"):
        if masks[name].dtype != bool or masks[name].shape != values.shape:
            raise ValueError(f"Invalid current-QC {name} mask")
    for channel in BELTS:
        if channel not in lm["belts"]:
            raise ValueError(f"Reviewed landmark channel missing: {channel}")
        belt = lm["belts"][channel]
        column = data["channels"].index(channel)
        if (
            belt["review_status"] != "manual_reviewed"
            or belt["channel_status"]
            != data["metadata"]["qc"]["channel_ratings"][channel]
            or not np.array_equal(
                belt["source_waveform"], values[:, column], equal_nan=True
            )
            or not np.array_equal(
                belt["detection_waveform"], values[:, column], equal_nan=True
            )
            or not np.array_equal(belt["sample_valid"], masks["valid"][:, column])
            or not np.array_equal(belt["artifact_mask"], masks["artifact"][:, column])
        ):
            raise ValueError(
                f"{channel}: landmark waveform/QC differs from preprocessing"
            )
    return data, lm, ref, provenance


def compatible_inputs(recording_id, preprocessing_root, landmark_root, reference_root):
    """Choose newest compatible reference and its exact hash-linked input files."""
    roots = {
        "preprocessing": Path(preprocessing_root).resolve(),
        "landmarks": Path(landmark_root).resolve(),
    }
    references = list(
        (Path(reference_root) / recording_id).glob(f"{recording_id}_references_r*.json")
    )
    candidates = []
    for path in references:
        ref = json.loads(path.read_text(encoding="utf-8"))
        if (
            ref["schema_version"] != REFERENCE_SCHEMA
            or ref["identity"]["recording_id"] != recording_id
        ):
            raise ValueError(f"Incompatible reference result: {path}")
        candidates.append((ref["revision"], path, ref))
    hash_cache = {}
    for _, reference_path, ref in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        selected = {}
        for name, root in roots.items():
            source = ref["provenance"][name]
            recorded = Path(source["path"]).resolve()
            paths = (
                [recorded]
                if recorded.is_relative_to(root) and recorded.is_file()
                else []
            )
            paths.extend(
                p for p in (root / recording_id).rglob("*.h5") if p not in paths
            )
            for path in paths:
                if path not in hash_cache:
                    hash_cache[path] = source_sha256(path)
                if hash_cache[path] == source["sha256"]:
                    selected[name] = path
                    break
        if len(selected) == 2:
            return selected["preprocessing"], selected["landmarks"], reference_path
        logger.warning(
            "Reference revision lacks compatible saved inputs: %s", reference_path
        )
    raise FileNotFoundError(
        f"{recording_id}: no compatible saved preprocessing/landmark/reference triplet"
    )


def save_features(result, output_dir):
    """Atomically publish features.h5 and summary.json in a new runN directory."""
    recording_id = result["identity"]["recording_id"]
    if re.fullmatch(r"[A-Za-z0-9_-]+", recording_id) is None:
        raise ValueError("Unsafe recording ID")
    folder = Path(output_dir).resolve() / recording_id
    folder.mkdir(parents=True, exist_ok=True)
    runs = [
        int(p.name[3:])
        for p in folder.glob("run[0-9]*")
        if p.is_dir() and p.name[3:].isdigit()
    ]
    destination = folder / f"run{max(runs, default=-1) + 1}"
    metadata = {k: v for k, v in result.items() if k != "features"}
    metadata.update(
        created_at_utc=datetime.now(UTC).isoformat(),
        result_path=str(destination / "features.h5"),
    )
    summary = dict(metadata, features={})
    with tempfile.TemporaryDirectory(prefix=".features-", dir=folder) as temporary:
        with h5py.File(Path(temporary) / "features.h5", "w") as handle:
            handle.attrs.update(
                schema_version=SCHEMA, recording_id=recording_id, index_base=0
            )
            handle.create_dataset(
                "metadata_json",
                data=json.dumps(metadata, allow_nan=False),
                dtype=h5py.string_dtype(),
            )
            for channel, features in result["features"].items():
                summary["features"][channel] = {}
                for name, feature in features.items():
                    group = handle.create_group(f"features/{channel}/{name}")
                    details = {}
                    for key, value in feature.items():
                        if isinstance(value, np.ndarray):
                            options = {"compression": "gzip"} if value.size else {}
                            group.create_dataset(key, data=value, **options)
                        else:
                            details[key] = value
                    group.create_dataset(
                        "metadata_json",
                        data=json.dumps(details, allow_nan=False),
                        dtype=h5py.string_dtype(),
                    )
                    summary["features"][channel][name] = {
                        "unit": feature["unit"],
                        "resolution": feature["resolution"],
                        "available": feature["available"],
                        "quality": feature["quality"],
                        "observations": len(feature["values"]),
                        "valid_observations": int(feature["valid"].sum()),
                    }
            handle.flush()
        with (Path(temporary) / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        # Every target run contains both files; an existing run is never replaced.
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    return destination / "features.h5"
