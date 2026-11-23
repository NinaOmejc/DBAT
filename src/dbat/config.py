"""Load the single grouped TOML configuration without a configuration framework."""

import hashlib
import math
from copy import deepcopy
from pathlib import Path

import tomllib

CHANNELS = ("ECG1", "ECG2", "SpO2", "Resp-Lungs", "Blood Pressure", "Resp-Diaphragm")
BELTS = ("Resp-Lungs", "Resp-Diaphragm")


def window_samples(value, unit, fs_hz):
    """Convert positive seconds or integer samples to MATLAB-rounded samples."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Window length must be a positive finite number")
    if not math.isfinite(value) or value <= 0 or unit not in {"seconds", "samples"}:
        raise ValueError("Window length must be positive; unit must be seconds or samples")
    if unit == "samples" and value != int(value):
        raise ValueError("A window expressed in samples must be an integer")
    samples = math.floor(value * fs_hz + 0.5) if unit == "seconds" else int(value)
    if samples < 1:
        raise ValueError("Window resolves to fewer than one sample at input.fs_hz")
    return samples


def load_config(path, overrides=None, *, explicit_dat=False):
    """Read and validate grouped settings; apply nested CLI overrides in memory.

    Relative paths use the TOML directory. An explicit .dat bypasses only the
    configured input-directory existence check; IO validates that file instead.
    Returns a dictionary with the resolved configuration path for provenance.
    """
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for group, changes in (overrides or {}).items():
        config[group].update(deepcopy(changes))
    try:
        _validate(config, path.parent, explicit_dat)
    except KeyError as error:
        raise ValueError(f"Missing configuration key: {error}; use configs/default.toml") from error
    config["config_path"] = str(path)
    config["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def _validate(config, directory, explicit_dat):
    """Reject unsupported settings and inconsistent units before loading data."""
    input_config, qc = config["input"], config["qc"]
    preprocessing, output = config["preprocessing"], config["output"]
    fs_hz = input_config["fs_hz"]
    if (
        isinstance(fs_hz, bool)
        or not isinstance(fs_hz, (int, float))
        or not math.isfinite(fs_hz)
        or fs_hz <= 0
    ):
        raise ValueError("input.fs_hz must be finite and positive (Hz)")
    if len(input_config["channels"]) != 6 or set(input_config["channels"]) != set(CHANNELS):
        raise ValueError(f"input.channels must list each of these six names once: {CHANNELS}")
    selected = preprocessing["channels"]
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(CHANNELS):
        raise ValueError("preprocessing.channels must contain unique input channel names")
    for name in ("filename_pattern", "alternate_filename_pattern"):
        pattern = input_config[name]
        if not pattern and name == "alternate_filename_pattern":
            continue
        try:
            filename = pattern.format(subject=1, measurement=1)
        except (KeyError, ValueError) as error:
            raise ValueError(f"input.{name}: use {{subject}} and {{measurement}} tokens") from error
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".dat":
            raise ValueError(f"input.{name} must resolve to a .dat filename without directories")
    for name, value in config["paths"].items():
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = directory / resolved
        config["paths"][name] = str(resolved.resolve())
    source_dir, output_dir = (Path(config["paths"][name]) for name in ("input_dir", "output_dir"))
    if not explicit_dat and not source_dir.is_dir():
        raise ValueError(
            f"Input directory does not exist: {source_dir}; set paths.input_dir or --dat"
        )
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("paths.output_dir must be outside paths.input_dir")
    parent = output_dir
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError(f"Output path has a file as an ancestor: {parent}")
    window_samples(qc["window_length"], qc["window_unit"], fs_hz)
    if qc["artifact_scope"] not in {"both_belts", "selected_channel"}:
        raise ValueError("qc.artifact_scope must be both_belts or selected_channel")
    if qc["initial_rating"] != "unassessed":
        raise ValueError("qc.initial_rating must be unassessed; ratings require explicit review")
    if not isinstance(qc["reviewer"], str):
        raise ValueError("qc.reviewer must be text")
    if type(qc["max_plot_points"]) is not int or qc["max_plot_points"] < 100:
        raise ValueError("qc.max_plot_points must be an integer >= 100")
    for value, name in [
        (qc["open_gui"], "qc.open_gui"),
        (qc["shared_belt_scale"], "qc.shared_belt_scale"),
        (preprocessing["enabled"], "preprocessing.enabled"),
        (output["save_preview"], "output.save_preview"),
    ]:
        if type(value) is not bool:
            raise ValueError(f"{name} must be true or false")
    smoothing, detrend = preprocessing["smoothing"], preprocessing["detrend"]
    if smoothing["method"] not in {"none", "moving_mean", "savitzky_golay"}:
        raise ValueError("smoothing.method must be none, moving_mean or savitzky_golay")
    window_samples(smoothing["window_sec"], "seconds", fs_hz)
    sg_window = window_samples(smoothing["sg_window"], smoothing["sg_window_unit"], fs_hz)
    if (
        sg_window % 2 != 1
        or type(smoothing["polyorder"]) is not int
        or not 0 <= smoothing["polyorder"] < sg_window
    ):
        raise ValueError(
            "Savitzky-Golay needs an odd window in samples and 0 <= polyorder < window"
        )
    if detrend["method"] not in {"none", "moving_detrend"}:
        raise ValueError(
            "detrend.method must be none or moving_detrend; hpfilter is not implemented"
        )
    if window_samples(detrend["window_sec"], "seconds", fs_hz) < 3:
        raise ValueError("detrend.window_sec must resolve to at least three samples")
    if preprocessing["artifacts"]["automatic_detection"] is not False:
        raise ValueError("Automatic artifact detection is unvalidated and not implemented")
    if output["schema_version"] != "dbat_preprocessing_v1" or output["overwrite"] is not False:
        raise ValueError("Use dbat_preprocessing_v1 and output.overwrite=false (versioned saves)")
    try:
        filename = output["filename"].format(recording_id="Sub1_M1")
    except (KeyError, ValueError) as error:
        raise ValueError("output.filename supports only {recording_id}") from error
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".h5":
        raise ValueError("output.filename must resolve to a .h5 filename without directories")
    for name in ("preview_start_sec", "preview_duration_sec"):
        value = output[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"output.{name} must be finite and nonnegative (seconds)")
    if output["preview_duration_sec"] == 0:
        raise ValueError("output.preview_duration_sec must be positive")
