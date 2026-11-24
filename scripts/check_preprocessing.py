"""Check preprocessing components, then inspect configured real recordings.

Run directly in PyCharm or use:
    python scripts/check_preprocessing.py --config configs/default.toml
    python scripts/check_preprocessing.py --self-checks-only
    python scripts/check_preprocessing.py --headless
Self-checks exercise all preprocessing modules on temporary synthetic recordings,
including GUI callbacks with an Agg canvas (no desktop interaction). Interactive
GUI appearance and physical mouse/keyboard use still require manual inspection.
Processes every selected subject/measurement pair, skipping explicit exclusions.
Writes versioned HDF5/JSON results and optional previews, then checks each round trip.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt

from dbat.config import BELTS, CHANNELS, RESULT_SCHEMA, load_config, window_samples
from dbat.io.recording import (
    load_dat_recording,
    load_preprocessing_result,
    load_qc_annotations,
    save_preprocessing_result,
    save_qc_annotations,
    source_sha256,
)
from dbat.logging import configure_logging, logger
from dbat.preprocessing.annotations import (
    annotation_masks,
    continuous_runs,
    make_interval,
    new_qc_annotations,
    recording_identity,
    review_status,
    utc_now,
    validate_qc_annotations,
)
from dbat.preprocessing.cleaning import (
    clean_recording,
    odd_samples,
    paired_breath_candidates,
    preprocess_belt_artifacts,
    preprocess_spo2,
    robust_scale,
)
from dbat.preprocessing.filtering import filter_belt
from dbat.preprocessing.pipeline import (
    prepare_recording,
    preprocess_belt,
    preprocess_bp,
    preprocess_ecg,
    preprocess_recording,
)
from dbat.preprocessing.polarity import (
    POLARITY_LABELS,
    apply_polarity,
    assess_relative_belt_polarity,
    assessment_text,
    saved_polarity_decision,
)
from dbat.preprocessing.resampling import resample_preprocessed
from dbat.preprocessing.smoothing import moving_mean, smooth_belt

REPOSITORY = Path(__file__).resolve().parents[1]


def _accepted_qc(recording):
    """Create explicitly synthetic review evidence for component checks only."""
    qc = new_qc_annotations(recording, "synthetic-check")
    qc["accepted_at_utc"] = utc_now()
    for name in recording.channels:
        qc["channel_ratings"][name] = "usable"
        qc["rating_provenance"][name] = {
            "reviewer": qc["reviewer"],
            "created_at_utc": qc["accepted_at_utc"],
        }
    qc["reviewed_intervals"] = [
        make_interval(
            0, len(recording.samples), recording.channels, recording.fs_hz, qc["reviewer"]
        )
    ]
    qc["review_status"] = review_status(qc)
    return qc


def check_annotations(recording, config, prepared):
    """Check native QC identity, coverage, artifact unions and independent JSON IO."""
    assert window_samples(0.25, "seconds", 200) == 50
    assert window_samples(51, "samples", 200) == 51
    with np.testing.assert_raises(ValueError):
        window_samples(0.25, "samples", 200)
    assert list(continuous_runs(np.array([False, True, True, False, True]))) == [(1, 3), (4, 5)]
    unreviewed = new_qc_annotations(recording)
    validate_qc_annotations(unreviewed, recording)
    assert not annotation_masks(unreviewed, recording)["reviewed"].any()
    assert recording_identity(recording) == unreviewed["identity"]
    qc = _accepted_qc(recording)
    qc["artifacts"] = [make_interval(200, 400, BELTS, recording.fs_hz, qc["reviewer"])]
    qc["review_status"] = review_status(qc)
    masks = annotation_masks(qc, recording)
    for name in BELTS:
        column = recording.channels.index(name)
        assert masks["artifact"][:, column].sum() == 200
        assert masks["reviewed"][:, column].all()
    assert not masks["artifact"][:, recording.channels.index("SpO2")].any()
    for field in ("identity", "artifacts"):
        invalid = deepcopy(qc)
        if field == "identity":
            invalid[field]["fs_hz"] += 1
        else:
            invalid[field][0]["end_sec"] += 1
        with np.testing.assert_raises(ValueError):
            validate_qc_annotations(invalid, recording)
    path = save_qc_annotations(qc, recording, config["paths"]["output_dir"])
    loaded = load_qc_annotations(path, recording)
    assert loaded["artifacts"] == qc["artifacts"] and loaded["loaded_from_file"] == str(path)
    with np.testing.assert_raises(ValueError):
        save_qc_annotations(unreviewed, recording, config["paths"]["output_dir"])
    with np.testing.assert_raises(ValueError):
        load_dat_recording(config, recording.source_path, subject=998, measurement=1)


def check_cleaning(recording, config, prepared):
    """Check brief correction, broad paired-breath protection, SpO2 and passthroughs."""
    fs = recording.fs_hz
    x = recording.samples[:, recording.channels.index(BELTS[0])].copy()
    assert np.isclose(robust_scale(np.array([0.0, 1.0, 2.0])), 1.4826)
    assert odd_samples(0.25, fs) == 51
    glitch = x.copy()
    glitch[8000] += 30
    corrected, interpolated, _, events, _ = preprocess_belt_artifacts(
        glitch, fs, np.array([], int), 0.25
    )
    assert interpolated[8000] and abs(corrected[8000] - x[8000]) < 0.01 and events
    protected = preprocess_belt_artifacts(glitch, fs, np.array([8000]), 0.25)
    assert protected[0][8000] == glitch[8000] and protected[2][8000]
    time = recording.time_sec
    broad = x + 15 * np.exp(-(((time - 50) / 1.2) ** 2) / 2)
    paired = paired_breath_candidates(dict.fromkeys(BELTS, broad), fs)
    assert np.any(np.abs(paired / fs - 50) < 1.5)
    negative = x.copy()
    negative[12000:12400] -= 20
    result = preprocess_belt_artifacts(negative, fs, np.array([], int), 0.25)
    assert result[2][12000:12400].any()
    flat = np.full(len(x), 97.0)
    drop = flat.copy()
    drop[6000:6400] = 65
    options = config["preprocessing"]["artifacts"]
    result = preprocess_spo2(drop, fs, options)
    assert result[1].any() and np.isfinite(result[0]).all()
    np.testing.assert_allclose(result[0][6100:6300], 97.0)
    unknown = preprocess_spo2(drop / 100, fs, {**options, "spo2_units": "unknown"})
    np.testing.assert_array_equal(unknown[0], drop / 100)
    assert not unknown[1].any()
    missing = np.full(100, np.nan)
    for result in (
        preprocess_spo2(missing, fs, options),
        preprocess_belt_artifacts(missing, fs, np.array([], int), 0.25),
    ):
        assert np.isnan(result[0]).all() and not result[1].any()
    disabled = deepcopy(config)
    disabled["preprocessing"]["artifacts"]["automatic_detection"] = False
    clean = clean_recording(recording, disabled)
    np.testing.assert_array_equal(clean["corrected"], recording.samples)
    assert not clean["masks"]["interpolated"].any()
    assert prepared["identity"] == recording_identity(recording)
    assert prepare_recording(recording, config)["signature"] == prepared["signature"]
    for passthrough in (preprocess_ecg, preprocess_bp):
        copied = passthrough(x)
        np.testing.assert_array_equal(copied, x)
        assert copied is not x


def check_smoothing_filters(recording, config, prepared):
    """Check all smoothers, full-series SOS filtering, gap support and detrending."""
    x = recording.samples[:, recording.channels.index(BELTS[0])]
    finite = np.ones(len(x), bool)
    eligible, influence = finite.copy(), np.zeros(len(x), bool)
    eligible[500:510], influence[1000] = False, True
    np.testing.assert_allclose(moving_mean(np.arange(7.0), 4), [0.5, 1, 1.5, 2.5, 3.5, 4.5, 5])
    for method, width in (("none", 1), ("moving_mean", 50), ("savitzky_golay", 51)):
        parameters = {"method": method, "window_samples": width, "polyorder": 3}
        result = smooth_belt(x, finite, eligible, influence, parameters)
        expected = (
            x
            if method == "none"
            else moving_mean(x, width)
            if method == "moving_mean"
            else savgol_filter(x, width, 3, mode="interp")
        )
        np.testing.assert_array_equal(result[0], expected)
        assert not result[2][500:510].any() and result[3][1000]
        gap = x.copy()
        gap[400:410] = np.nan
        result = smooth_belt(gap, np.isfinite(gap), np.isfinite(gap), influence, parameters)
        assert np.isnan(result[0][400:410]).all() and np.isfinite(result[0][410:]).all()
    short = smooth_belt(
        x[:5],
        finite[:5],
        finite[:5],
        influence[:5],
        {"method": "savitzky_golay", "window_samples": 7, "polyorder": 3},
    )
    assert not short[1].any() and len(short[4]) == 1
    edge_eligible = finite.copy()
    edge_eligible[49] = False
    sg_edge = smooth_belt(
        x,
        finite,
        edge_eligible,
        influence,
        {"method": "savitzky_golay", "window_samples": 51, "polyorder": 3},
    )
    assert not sg_edge[2][:25].any()
    filters = {
        "highpass": {"enabled": True, "cutoff_hz": 0.01, "edge_pad_sec": 100.0},
        "lowpass": {"enabled": True, "cutoff_hz": 20.0, "edge_pad_sec": 1.0},
    }
    actual = filter_belt(x, eligible, influence, recording.fs_hz, filters)
    expected = x.copy()
    for name, options in filters.items():
        pad = min(max(12, round(options["edge_pad_sec"] * recording.fs_hz)), (len(x) - 1) // 2)
        sos = butter(4, options["cutoff_hz"], btype=name, fs=recording.fs_hz, output="sos")
        expected = sosfiltfilt(
            sos, np.pad(expected, pad, mode="reflect"), padtype="odd", padlen=12
        )[pad : pad + len(x)]
        assert len(actual[3][name]["runs"]) == 1
    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(
        actual[0], filter_belt(x, finite, influence, recording.fs_hz, filters)[0]
    )
    assert not actual[1][500:510].any()
    disabled = {name: {**options, "enabled": False} for name, options in filters.items()}
    np.testing.assert_array_equal(
        filter_belt(x, finite, influence, recording.fs_hz, disabled)[0], x
    )
    short = filter_belt(x[:8], finite[:8], influence[:8], recording.fs_hz, filters)
    assert not short[1].any() and not short[3]["highpass"]["applied"]
    gap = x.copy()
    gap[400:410] = np.nan
    filtered_gap = filter_belt(gap, np.isfinite(gap), influence, recording.fs_hz, filters)
    assert np.isnan(filtered_gap[0][400:410]).all() and np.isfinite(filtered_gap[0][410:]).all()
    assert len(filtered_gap[3]["highpass"]["runs"]) == 2
    missing = np.full(100, np.nan)
    filtered_missing = filter_belt(
        missing,
        np.zeros(100, bool),
        np.zeros(100, bool),
        recording.fs_hz,
        filters,
    )
    assert np.isnan(filtered_missing[0]).all() and not filtered_missing[1].any()
    result = preprocess_belt(
        x,
        ~finite,
        influence,
        recording.fs_hz,
        {"method": "moving_detrend", "window_samples": 101},
        disabled,
    )
    expected = x - moving_mean(x, 101)
    np.testing.assert_array_equal(result[0], expected - expected.mean())


def check_polarity(recording, config, prepared):
    """Check disabled/insufficient/positive/negative diagnostics and explicit flips."""
    values, fs, names = recording.samples, recording.fs_hz, recording.channels
    reliable = np.ones(values.shape, bool)
    disabled = assess_relative_belt_polarity(values, names, fs, reliable, False)
    assert assessment_text(disabled) == "Not assessed"
    insufficient = assess_relative_belt_polarity(values, names, fs, ~reliable, True)
    assert assessment_text(insufficient) == "Insufficient reliable data"
    positive = assess_relative_belt_polarity(values, names, fs, reliable, True)
    assert positive["status"] == "assessed" and not positive["polarity_warning"]
    inverted = apply_polarity(values, names, BELTS[1])
    negative = assess_relative_belt_polarity(inverted, names, fs, reliable, True)
    assert negative["polarity_warning"] and "Check polarity" in assessment_text(negative)
    np.testing.assert_array_equal(apply_polarity(inverted, names, BELTS[1]), values)
    np.testing.assert_array_equal(apply_polarity(values, names, None), values)
    with np.testing.assert_raises(ValueError):
        apply_polarity(values, names, "both")
    assert saved_polarity_decision({}, prepared) is None
    decision = {"preparation_signature": prepared["signature"]}
    assert saved_polarity_decision({"polarity": decision}, prepared) == decision
    with np.testing.assert_raises(ValueError):
        saved_polarity_decision({"polarity": {"preparation_signature": "0" * 64}}, prepared)


def check_pipeline_storage(recording, config, prepared):
    """Check QC independence, optional grids/methods, previews and numbered storage."""
    qc = _accepted_qc(recording)
    automatic = preprocess_recording(recording, config, prepared=prepared)
    reviewed = preprocess_recording(recording, config, qc, prepared=prepared)
    np.testing.assert_array_equal(automatic.smoothed, reviewed.smoothed)
    assert not automatic.masks["valid"].any() and reviewed.masks["valid"].any()
    assert (
        reviewed.original is recording.samples and reviewed.respiratory_signal is reviewed.smoothed
    )
    reviewed.require_native_grid()
    for method in ("none", "savitzky_golay"):
        variant = deepcopy(config)
        variant["preprocessing"]["smoothing"]["method"] = method
        result = preprocess_recording(recording, variant, qc, prepared=prepared)
        for name in BELTS:
            j = recording.channels.index(name)
            expected = (
                result.preprocessed[:, j]
                if method == "none"
                else savgol_filter(result.preprocessed[:, j], 51, 3, mode="interp")
            )
            np.testing.assert_array_equal(result.smoothed[:, j], expected)
    for rate in (100.0, 300.0):
        variant = deepcopy(config)
        variant["preprocessing"]["resampling"]["resample_freq"] = rate
        result = preprocess_recording(recording, variant, qc, prepared=prepared)
        assert result.processed_fs_hz == rate and len(result.smoothed) == round(
            len(recording.samples) * rate / recording.fs_hz
        )
        assert np.isfinite(result.smoothed).all() and result.masks["boundary"].any()
        with np.testing.assert_raises(ValueError):
            result.require_native_grid()
    for rate in (0.0, recording.fs_hz):
        values, _, _, _, info = resample_preprocessed(
            reviewed.preprocessed,
            reviewed.masks,
            recording.fs_hz,
            {**config["preprocessing"]["resampling"], "resample_freq": rate},
        )
        assert values is reviewed.preprocessed and not info["applied"]
    missing = reviewed.preprocessed.copy()
    missing[1000:1010, 3], missing[:, 4] = np.nan, np.nan
    masks = {name: value.copy() for name, value in reviewed.masks.items()}
    masks["finite"] = np.isfinite(missing)
    resampled = resample_preprocessed(
        missing,
        masks,
        recording.fs_hz,
        {**config["preprocessing"]["resampling"], "resample_freq": 100.0},
    )
    assert np.isnan(resampled[0][:, 4]).all() and resampled[1]["missing_influence"].any()
    first = save_preprocessing_result(reviewed, config)
    verify_saved_result(first, reviewed)
    assert first.parent.name == "run0" and (first.parent / "preview.png").is_file()
    loaded = load_preprocessing_result(first)
    np.testing.assert_array_equal(loaded["signals"]["smoothed"], reviewed.smoothed)
    config = deepcopy(config)
    config["output"]["save_preview"] = False
    second = save_preprocessing_result(reviewed, config)
    assert second.parent.name == "run1" and not (first.parent.parent / "runs").exists()
    (first.parent.parent / ".run2.partial").mkdir()
    third = save_preprocessing_result(reviewed, config)
    assert third.parent.name == "run3" and first.is_file()
    with np.testing.assert_raises(ValueError):
        load_preprocessing_result(first.parent)
    disabled = deepcopy(config)
    disabled["preprocessing"]["enabled"] = False
    bypass = preprocess_recording(recording, disabled, qc, prepared=prepared)
    np.testing.assert_array_equal(bypass.smoothed, prepared["corrected"])


def check_gui_callbacks(recording, config, prepared):
    """Exercise GUI navigation, editing, save and cancel without desktop interaction."""
    import matplotlib
    import matplotlib.pyplot as plt

    from dbat.preprocessing.preproc_gui import ManualQCEditor, run_qc_gui

    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    try:
        variant = deepcopy(config)
        variant["preprocessing"]["check_relative_belt_polarity"] = True
        variant["qc"].update(window_length=10.0, max_plot_points=200, reviewer="synthetic-check")
        prepared = prepare_recording(recording, variant)
        original = new_qc_annotations(recording)
        with np.testing.assert_raises(RuntimeError):
            run_qc_gui(recording, variant, original, prepared)
        editor = ManualQCEditor(recording, variant, original, prepared)
        editor.reviewer.set_val("")
        editor._set_rating(BELTS[0], "usable")
        assert editor.qc["channel_ratings"][BELTS[0]] == "unassessed"
        editor.reviewer.set_val("synthetic-check")
        editor._move(1)
        editor._zoom(0.5)
        editor._set_window("5")
        editor._navigate(0)
        for _ in range(3):
            editor._toggle_display(None)
        editor._toggle_polarity_menu(None)
        for label in (POLARITY_LABELS[1], POLARITY_LABELS[2], POLARITY_LABELS[0]):
            editor._select_polarity(label)
        np.testing.assert_array_equal(editor.corrected, prepared["corrected"])
        editor._add_artifact(1, 2, BELTS[0])
        assert set(editor.qc["artifacts"][0]["channels"]) == set(BELTS)
        editor._remove_artifact(SimpleNamespace(button=3, inaxes=editor.axes[0], xdata=1.5))
        assert not editor.qc["artifacts"]
        editor.scope.set_active(1)
        editor._add_artifact(2, 3, "SpO2")
        assert editor.qc["artifacts"][0]["channels"] == ["SpO2"]
        for name in variant["qc"]["channels"]:
            editor._set_rating(name, "usable")
        editor._confirm_visible(None)
        editor._save(None)
        assert editor.result is not None
        validate_qc_annotations(editor.result, recording)
        assert not original["artifacts"] and original["accepted_at_utc"] is None
        cancelled = ManualQCEditor(recording, variant, original, prepared)
        cancelled._add_artifact(1, 2, BELTS[0])
        cancelled._cancel(None)
        assert cancelled.result is None and not original["artifacts"]
    finally:
        plt.close("all")
        plt.switch_backend(previous)


def check_preprocessing_functions(config):
    """Run synthetic checks for every preprocessing component; leave no output files."""
    checks = (
        check_annotations,
        check_cleaning,
        check_smoothing_filters,
        check_polarity,
        check_pipeline_storage,
        check_gui_callbacks,
    )
    with TemporaryDirectory(prefix="dbat-preprocessing-check-") as directory:
        folder = Path(directory)
        (folder / "source").mkdir()
        local = deepcopy(config)
        local["input"].update(fs_hz=200.0, channels=list(CHANNELS))
        local["paths"]["output_dir"] = str(folder / "outputs")
        local["qc"].update(channels=[*BELTS, "SpO2"], open_gui=False)
        local["output"].update(save_preview=True, preview_duration_sec=10.0)
        local["preprocessing"].update(enabled=True, check_relative_belt_polarity=False)
        local["preprocessing"]["artifacts"]["automatic_detection"] = True
        local["preprocessing"]["detrend"]["method"] = "none"
        local["preprocessing"]["resampling"]["resample_freq"] = 0.0
        local["preprocessing"]["smoothing"].update(
            method="moving_mean",
            window_sec=0.25,
            sg_window=51.0,
            sg_window_unit="samples",
            polyorder=3,
        )
        for options in local["preprocessing"]["filtering"].values():
            options["enabled"] = False
        time = np.arange(36000) / 200.0
        belt = np.cos(2 * np.pi * 0.25 * time)
        values = np.column_stack(
            [
                np.sin(2 * np.pi * time),
                np.cos(2 * np.pi * time),
                np.full(time.size, 97.0),
                belt,
                120 + 5 * belt,
                belt,
            ]
        )
        path = folder / "source" / "synthetic_fs200_Sub999_M1.dat"
        np.savetxt(path, values, fmt="%.12g")
        recording = load_dat_recording(local, path)
        prepared = prepare_recording(recording, local)
        for check in checks:
            logger.info("Checking %s...", check.__name__.removeprefix("check_"))
            check(recording, local, prepared)
            logger.info("  PASS")
    return [check.__name__ for check in checks]


def verify_saved_result(path, result):
    """Reopen output and check native originals, primary grids, masks and metadata."""
    recording = result.recording
    logger.info("  Verifying saved result for %s...", recording.recording_id)
    with (path.parent / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    expected_arrays = {
        "signals/original": result.original,
        "signals/preprocessed": result.preprocessed,
        "signals/smoothed": result.smoothed,
        "original_time_sec": result.original_time_sec,
        "processed_time_sec": result.processed_time_sec,
        **{f"masks/{name}": mask for name, mask in result.masks.items()},
    }
    with h5py.File(path, "r") as handle:
        metadata = json.loads(handle["metadata_json"].asstr()[()])
        if (
            metadata != summary
            or metadata["schema_version"] != RESULT_SCHEMA
            or handle.attrs["schema_version"] != RESULT_SCHEMA
        ):
            raise ValueError("HDF5 and JSON metadata/schema disagree")
        if (
            handle.attrs["original_fs_hz"] != result.original_fs_hz
            or handle.attrs["processed_fs_hz"] != result.processed_fs_hz
            or handle["channels"].asstr()[:].tolist() != list(recording.channels)
        ):
            raise ValueError("Sampling rates or channel order changed during saving")
        if (
            set(handle["signals"]) != {"original", "preprocessed", "smoothed"}
            or "resampled" in handle
        ):
            raise ValueError(
                "Expected exactly three canonical signals and no duplicate resampled bundle"
            )
        for name, expected in expected_arrays.items():
            saved = handle[name][:]
            if (
                saved.dtype != expected.dtype
                or saved.shape != expected.shape
                or not np.array_equal(saved, expected, equal_nan=True)
            ):
                raise ValueError(f"Round-trip data mismatch: {name}")
        for name in ("original", "preprocessed", "smoothed"):
            expected_fs = result.original_fs_hz if name == "original" else result.processed_fs_hz
            expected_axis = "original_time_sec" if name == "original" else "processed_time_sec"
            if (
                handle[f"signals/{name}"].attrs["fs_hz"] != expected_fs
                or handle[f"signals/{name}"].attrs["time_axis"] != expected_axis
            ):
                raise ValueError(f"Signal sampling metadata disagrees: {name}")
        if metadata["qc"] != result.qc:
            raise ValueError("Saved QC annotations differ from the accepted input")
        validate_qc_annotations(metadata["qc"], recording)
    if not np.array_equal(result.original, recording.samples, equal_nan=True):
        raise ValueError("Original samples changed")
    n_samples = len(result.preprocessed)
    if result.smoothed.shape != (n_samples, len(recording.channels)):
        raise ValueError("Primary processed signal shapes disagree")
    for times, fs_hz, count in (
        (result.original_time_sec, result.original_fs_hz, len(result.original)),
        (result.processed_time_sec, result.processed_fs_hz, n_samples),
    ):
        if not np.array_equal(times, np.arange(count) / fs_hz):
            raise ValueError("Signal time grid disagrees with its zero origin and sampling rate")
    rate = result.provenance["resampling"]
    if rate["applied"]:
        expected_count = (len(result.original) * rate["up"] + rate["down"] - 1) // rate["down"]
        if n_samples != expected_count:
            raise ValueError("Primary processed sample count disagrees with the resampling ratio")
        numerator = np.arange(n_samples) * rate["down"]
        left = np.minimum(numerator // rate["up"], len(result.original) - 1)
        right = np.minimum((numerator + rate["up"] - 1) // rate["up"], len(result.original) - 1)
        available = np.isfinite(result.original[left]) & np.isfinite(result.original[right])
    else:
        result.require_native_grid()
        available = np.isfinite(result.original)
    for values in (result.preprocessed, result.smoothed):
        if np.any(available & ~np.isfinite(values)) or np.any(~available & np.isfinite(values)):
            raise ValueError("Numerical availability disagrees with genuine source support")
    for name, mask in result.masks.items():
        if mask.dtype != bool or mask.shape != result.smoothed.shape:
            raise ValueError(f"Mask is not Boolean on the primary processed grid: {name}")
    if np.any(
        result.masks["valid"]
        & (~result.masks["reviewed"] | result.masks["artifact"] | ~result.masks["smoothed_support"])
    ):
        raise ValueError("Processing accepted unreviewed, artifact or unsupported samples")
    if rate["applied"] and np.any(
        result.masks["valid"] & (result.masks["boundary"] | result.masks["missing_influence"])
    ):
        raise ValueError("Processing accepted contaminated FIR support")
    if (
        result.respiratory_signal is not result.smoothed
        or result.provenance["respiratory_analysis_fs_hz"] != result.processed_fs_hz
    ):
        raise ValueError("Respiratory analysis must use smoothed and its processed sampling rate")
    for column, name in enumerate(recording.channels):
        if (
            name not in BELTS
            or name not in result.provenance["selected_channels"]
            or result.provenance["smoothing"]["method"] == "none"
        ):
            if not np.array_equal(
                result.smoothed[:, column], result.preprocessed[:, column], equal_nan=True
            ):
                raise ValueError(
                    "A channel without respiratory smoothing differs from preprocessed"
                )
    if source_sha256(recording.source_path) != recording.source_sha256:
        raise ValueError("Source .dat checksum changed since import")
    with (path.parent / "validation.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "passed",
                "checked_at_utc": utc_now(),
                "schema_version": "dbat_manual_check_v3",
                "recording_id": recording.recording_id,
                "source_sha256": recording.source_sha256,
                "checks": [
                    "original samples/checksum unchanged",
                    "three canonical signals; no duplicate resampled bundle",
                    "explicit original and processed sampling rates/time axes",
                    "processed-grid Boolean masks",
                    "numerical availability independent of QC/support masks",
                    "smoothed respiratory source and processed rate",
                    "QC identity/bounds/provenance",
                    "exact HDF5/JSON round trip",
                ],
            },
            stream,
            indent=2,
        )
    logger.debug(
        "Round trip verified: %s | original=%s at %g Hz | processed=%s at %g Hz",
        recording.recording_id,
        result.original.shape,
        result.original_fs_hz,
        result.smoothed.shape,
        result.processed_fs_hz,
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
        from dbat.preprocessing.preproc_gui import run_qc_gui

        qc = run_qc_gui(recording, config, qc, prepared=prepared)
        if qc is None:
            return None
        save_qc_annotations(qc, recording, config["paths"]["output_dir"])

    result = preprocess_recording(recording, config, qc, prepared=prepared)
    path = save_preprocessing_result(result, config)

    verify_saved_result(path, result)
    return path


def main(config_file, *, self_checks_only=False, headless=False, output_dir=None):
    """Run Phase 1 for the configured subject/measurement Cartesian product.

    Use project logging configured from run.verbosity.

    Parameters
    ----------
    config_file : path-like
        Grouped TOML containing selection, paths, QC and processing settings.
    self_checks_only, headless : bool
        Run only temporary synthetic checks, or suppress real-recording GUI review.
        Headless execution creates no new manual QC evidence.
    output_dir : path-like, optional
        Explicit destination for real-recording check outputs.

    Returns
    -------
    list of Path
        Completed HDF5 outputs. Excluded and GUI-cancelled recordings are skipped;
        input or processing errors stop the run with the original error.
    """
    config = load_config(config_file)
    selection = config["run"]
    configure_logging(selection["verbosity"])
    check_preprocessing_functions(config)
    logger.info("All preprocessing component checks passed.")
    if self_checks_only:
        return []
    if headless:
        config["qc"]["open_gui"] = False
    if output_dir is not None:
        config["paths"]["output_dir"] = str(Path(output_dir).resolve())
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
    parser.add_argument(
        "--self-checks-only",
        action="store_true",
        help="Check every component using temporary synthetic data only",
    )
    parser.add_argument("--headless", action="store_true", help="Disable real-recording GUI review")
    parser.add_argument(
        "--output-dir", type=Path, help="Destination for real-recording check outputs"
    )
    args = parser.parse_args()
    main(
        args.config,
        self_checks_only=args.self_checks_only,
        headless=args.headless,
        output_dir=args.output_dir,
    )
