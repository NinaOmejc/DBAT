"""Load the single grouped TOML configuration without a configuration framework."""

import hashlib
import math
import re
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


def load_config(path):
    """Read one grouped TOML and validate selection, paths and scientific settings.

    Relative paths use the TOML directory. Returns the effective configuration
    plus the source file path/hash for provenance; no CLI overrides are applied.
    """
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    try:
        _validate(config, path.parent)
    except KeyError as error:
        raise ValueError(f"Missing configuration key: {error}; use configs/default.toml") from error
    config["config_path"] = str(path)
    config["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def _resolve_path(value, directory):
    """Resolve a configured path relative to its TOML file's directory."""
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else directory / path).resolve())


def _validate_selection(selection):
    """Require unique positive subject/measurement lists and excluded pairs."""
    for name in ("subjects", "measurements"):
        values = selection[name]
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value <= 0 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"run.{name} must be a nonempty list of unique positive integers")
    excluded = selection["exclude_recordings"]
    if not isinstance(excluded, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(type(value) is not int or value <= 0 for value in pair)
        for pair in excluded
    ):
        raise ValueError("run.exclude_recordings must be a list of [subject, measurement] pairs")
    if len({tuple(pair) for pair in excluded}) != len(excluded):
        raise ValueError("run.exclude_recordings contains duplicate pairs")


def _validate(config, directory):
    """Reject unsupported settings and inconsistent units before loading data."""
    _validate_selection(config["run"])
    verbosity = config["run"]["verbosity"]
    if type(verbosity) is not int or verbosity not in (0, 1, 2):
        raise ValueError("run.verbosity must be 0 (silent), 1 (progress), or 2 (detailed)")
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
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(BELTS):
        raise ValueError("preprocessing.channels must contain unique respiratory belt channel names")
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
        config["paths"][name] = _resolve_path(value, directory)
    source_dir, output_dir = (Path(config["paths"][name]) for name in ("input_dir", "output_dir"))
    dat_file = input_config["dat_file"]
    if not isinstance(dat_file, str):
        raise ValueError("input.dat_file must be a path string, or empty for filename resolution")
    if dat_file:
        if len(config["run"]["subjects"]) != 1 or len(config["run"]["measurements"]) != 1:
            raise ValueError("input.dat_file requires exactly one subject and one measurement")
        input_config["dat_file"] = _resolve_path(dat_file, directory)
        explicit_path = Path(input_config["dat_file"])
        if not explicit_path.is_file() or explicit_path.suffix.lower() != ".dat":
            raise ValueError(f"input.dat_file must identify an existing .dat file: {explicit_path}")
    elif not source_dir.is_dir():
        raise ValueError(
            f"Input directory does not exist: {source_dir}; set paths.input_dir or input.dat_file"
        )
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("paths.output_dir must be outside paths.input_dir")
    parent = output_dir
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError(f"Output path has a file as an ancestor: {parent}")
    qc_channels = qc["channels"]
    if (
        not isinstance(qc_channels, list)
        or not qc_channels
        or any(not isinstance(name, str) or name not in input_config["channels"] for name in qc_channels)
        or len(set(qc_channels)) != len(qc_channels)
    ):
        raise ValueError("qc.channels must be a nonempty list of unique input channel names")
    window_samples(qc["window_length"], qc["window_unit"], fs_hz)
    if qc["artifact_scope"] not in {"both_belts", "selected_channel"}:
        raise ValueError("qc.artifact_scope must be both_belts or selected_channel")
    if qc["initial_rating"] != "unassessed":
        raise ValueError("qc.initial_rating must be unassessed; ratings require explicit review")
    if not isinstance(qc["reviewer"], str):
        raise ValueError("qc.reviewer must be text")
    annotation_files = qc["annotation_files"]
    if not isinstance(annotation_files, dict):
        raise ValueError("qc.annotation_files must map recording IDs to accepted QC JSON files")
    for recording_id, filename in annotation_files.items():
        if (
            not re.fullmatch(r"Sub[1-9]\d*_M[1-9]\d*", recording_id)
            or not isinstance(filename, str)
            or not filename
        ):
            raise ValueError("qc.annotation_files needs SubN_MN keys and nonempty JSON paths")
        resolved = Path(_resolve_path(filename, directory))
        if not resolved.is_file() or resolved.suffix.lower() != ".json":
            raise ValueError(f"Accepted QC JSON does not exist: {resolved}")
        annotation_files[recording_id] = str(resolved)
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
    resampling = preprocessing["resampling"]
    frequency = resampling["resample_freq"]
    if (
        isinstance(frequency, bool)
        or not isinstance(frequency, (int, float))
        or not math.isfinite(frequency)
        or frequency < 0
    ):
        raise ValueError("resampling.resample_freq must be finite and nonnegative (Hz); 0 disables it")
    if type(resampling["filter_half_width"]) is not int or resampling["filter_half_width"] < 1:
        raise ValueError("resampling.filter_half_width must be a positive integer")
    beta = resampling["kaiser_beta"]
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0:
        raise ValueError("resampling.kaiser_beta must be finite and nonnegative")
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
