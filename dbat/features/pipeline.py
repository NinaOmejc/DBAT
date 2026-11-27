"""Single recording-level entry point for common physiological features."""

import copy
import importlib.metadata
from pathlib import Path

import numpy as np

from ..config import BELTS, validate_feature_config
from .catalog import REGISTRY, SUPPORTED_CHANNELS, select_features
from .catch22 import catch22_measurements
from .common import channel_signal, normalized_belt
from .io import SCHEMA, load_inputs, save_features
from .respiratory import (
    breath_measurements,
    eami_measurement,
    envelope_measurements,
    local_measurements,
    spectral_measurements,
)
from .spo2 import saturation_measurements
from .thoracoabdominal import (
    balance_measurements,
    coherence_measurement,
    phase_measurements,
)


def _analysis_intervals(bounds, count, fs):
    """Resolve requested elapsed-second bounds on the exact processed grid."""
    intervals = []
    for start, end in bounds or [(0, count / fs)]:
        first, stop = (
            int(np.floor(start * fs + 0.5)),
            min(count, int(np.floor(end * fs + 0.5))),
        )
        if first >= stop:
            raise ValueError("Analysis interval contains no recording samples")
        intervals.append(
            {
                "requested_start_sec": start,
                "requested_end_sec": end,
                "start_idx": first,
                "end_idx": stop,
                "start_sec": first / fs,
                "end_sec": stop / fs,
                "truncated": end > count / fs,
            }
        )
    return intervals


def _slice_observations(feature, intervals):
    """Keep observations with complete support; do not interpolate feature values."""
    indices, interval_ids = [], []
    for i, interval in enumerate(intervals):
        keep = (feature["time_sec"] >= interval["start_sec"]) & (
            feature["time_sec"] < interval["end_sec"]
        )
        contained = (feature["support_start_sec"] >= interval["start_sec"]) & (
            feature["support_end_sec"] <= interval["end_sec"]
        )
        if feature["resolution"] in {"breath", "inter_breath"}:
            contained &= feature["support_end_sec"] < interval["end_sec"]
        keep &= contained | ~feature["valid"]
        selected = np.flatnonzero(keep)
        indices.extend(selected)
        interval_ids.extend([i] * len(selected))
    indices = np.asarray(indices, int)
    count = len(feature["values"])
    out = {
        k: v[indices] if isinstance(v, np.ndarray) and len(v) == count else v
        for k, v in feature.items()
    }
    out["analysis_interval_index"] = np.asarray(interval_ids, int)
    return out


def calculate_features(
    data, *, landmarks, references, channels=None, features=None, config=None
):
    """Calculate selected features once and save a separate immutable recording run.

    Parameters
    ----------
    data, landmarks, references : path-like or dict
        Three matching saved Stage 3/4/5a artifacts, or unchanged loaded bundles.
        Loaded bundles require their recorded source files for hash validation.
    channels, features : sequence of str, optional
        Supported channel names and registry names/families. None uses [features]
        selection; defaults cover both belts, SpO2 and all 22 Catch22 descriptors.
    config : dict
        DBAT configuration with [features]; relative output paths use config_path's
        directory, or this project's configs/ directory when config_path is absent.

    Returns
    -------
    dict
        Features at natural temporal resolutions, analysis intervals, units,
        validity/support, exact input provenance and the saved features.h5 path.
    """
    if config is None:
        raise ValueError("Supply the DBAT configuration containing [features]")
    cfg = copy.deepcopy(config["features"])
    validate_feature_config(cfg)
    selected = select_features(cfg["features"] if features is None else features)
    selected_channels = list(cfg["channels"] if channels is None else channels)
    selected_channels = selected_channels or list(SUPPORTED_CHANNELS)
    if len(set(selected_channels)) != len(selected_channels) or not set(
        selected_channels
    ) <= set(SUPPORTED_CHANNELS):
        raise ValueError("Only respiratory belts and SpO2 have established Stage 5b QC")
    data, lm, refs, provenance = load_inputs(data, landmarks, references)
    fs = data["processed_fs_hz"]
    intervals = _analysis_intervals(
        cfg["analysis_intervals"], len(data["processed_time_sec"]), fs
    )
    result = {
        "schema_version": SCHEMA,
        "identity": copy.deepcopy(data["metadata"]["identity"]),
        "processed_fs_hz": fs,
        "signal_source": "preproc_signal",
        "index_convention": "zero_based_half_open",
        "settings": cfg,
        "analysis_intervals": intervals,
        "selected_channels": selected_channels,
        "selected_features": selected,
        "reference_settings": refs["settings"],
        "provenance": provenance,
        "features": {},
        "runtime": {
            n: importlib.metadata.version(n) for n in ("numpy", "scipy", "h5py")
        },
    }
    signals = {channel: channel_signal(data, channel) for channel in selected_channels}
    for channel in selected_channels:
        applicable = {n for n in selected if channel in REGISTRY[n]["channels"]}
        groups = {REGISTRY[n]["group"] for n in applicable}
        x, valid = signals[channel]
        out = {}
        if channel in BELTS:
            belt, reference = lm["belts"][channel], refs["respiratory"][channel]
            if "breaths" in groups:
                out.update(
                    {
                        n: v
                        for n, v in breath_measurements(belt, reference).items()
                        if n in applicable
                    }
                )
            if "local" in groups:
                out.update(
                    local_measurements(
                        x, valid, fs, belt, reference, cfg["respiratory"], applicable
                    )
                )
            if "envelope" in groups or "catch22" in groups:
                normalized = normalized_belt(x)
            if "envelope" in groups:
                out.update(
                    envelope_measurements(
                        normalized, valid, fs, cfg["respiratory"], applicable
                    )
                )
            for group, section in (("spectral", "spectral"), ("eami", "eami")):
                if group in groups and cfg[section]["band_hz"][1] >= fs / 2:
                    raise ValueError(
                        f"features.{section}.band_hz must be below processed Nyquist"
                    )
            if "spectral" in groups:
                out.update(
                    spectral_measurements(
                        x, valid, fs, belt, cfg["spectral"], applicable
                    )
                )
            if "eami" in groups:
                out["eami"] = eami_measurement(x, valid, fs, cfg["eami"])
        elif "spo2" in groups:
            out.update(saturation_measurements(x, fs, refs["spo2"], applicable))
        for name, feature in list(out.items()):
            out[name] = _slice_observations(feature, intervals)
        if "catch22" in groups:
            result["runtime"]["aeon"] = importlib.metadata.version("aeon")
            catch_names = [
                n
                for n in selected
                if n in applicable and REGISTRY[n]["group"] == "catch22"
            ]
            out.update(
                catch22_measurements(
                    normalized if channel in BELTS else x,
                    valid,
                    fs,
                    catch_names,
                    intervals,
                    cfg["catch22"],
                )
            )
            for name in catch_names:
                out[name]["analysis_interval_index"] = np.arange(len(intervals))
        result["features"][channel] = out
    joint = {n for n in selected if REGISTRY[n]["channels"] == ("two_belt",)}
    if joint and set(BELTS) <= set(selected_channels):
        x = [signals[c][0] for c in BELTS]
        valid = signals[BELTS[0]][1] & signals[BELTS[1]][1]
        groups = {REGISTRY[n]["group"] for n in joint}
        out = {}
        if "balance" in groups:
            out.update(
                balance_measurements(
                    valid,
                    fs,
                    lm["belts"],
                    refs["respiratory"],
                    cfg["thoracoabdominal"],
                    joint,
                )
            )
        for group in ("phase", "coherence"):
            if group in groups and cfg[group]["band_hz"][1] >= fs / 2:
                raise ValueError(
                    f"features.{group}.band_hz must be below processed Nyquist"
                )
        if "phase" in groups:
            out.update(
                phase_measurements(
                    x,
                    valid,
                    fs,
                    lm["belts"],
                    cfg["phase"],
                    cfg["thoracoabdominal"]["step_sec"],
                    joint,
                )
            )
        if "coherence" in groups:
            out["spectral_coherence"] = coherence_measurement(
                x, valid, fs, cfg["coherence"]
            )
        result["features"]["two_belt"] = {
            n: _slice_observations(v, intervals) for n, v in out.items()
        }
    elif joint:
        result["omitted_joint_features"] = {
            name: "both respiratory channels must be selected" for name in sorted(joint)
        }
    for channel, channel_features in result["features"].items():
        for name, feature in channel_features.items():
            feature.update(
                unit=REGISTRY[name]["unit"],
                dependencies=list(REGISTRY[name]["dependencies"]),
            )
            feature["available"] = bool(feature["valid"].any())
            feature["quality"] = (
                "good"
                if len(feature["valid"]) and feature["valid"].all()
                else ("partial" if feature["available"] else "unavailable")
            )
    base = Path(
        config.get(
            "config_path",
            Path(__file__).resolve().parents[2] / "configs" / "default.toml",
        )
    ).parent
    output_dir = Path(cfg["output_dir"])
    output_dir = output_dir if output_dir.is_absolute() else base / output_dir
    result["output_path"] = str(save_features(result, output_dir))
    return result
