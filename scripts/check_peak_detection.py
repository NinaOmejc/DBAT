"""Check the independent respiratory peak stage, optionally on saved preprocessing.

Run directly in PyCharm for synthetic checks, or use:
    python scripts/check_peak_detection.py
    python scripts/check_peak_detection.py --h5 outputs/preproc_data/Sub105_M1/run0/Sub105_M1_preprocessing.h5

Reads signals/smoothed on processed_fs_hz; never rewrites source files, smooths
again, accepts manual QC or recalculates features/labels. A failed check raises.
"""

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import find_peaks

from dbat.config import BELTS
from dbat.io.recording import load_preprocessing_result, source_sha256
from dbat.logging import configure_logging, logger
from dbat.peak_detection import (
    DetectorSettings,
    detect_belt,
    detect_peaks,
    detect_preprocessed,
    edit_peak,
    move_trough,
    peak_qc,
    recompute,
    validate_belt,
)


def check_automatic_detection():
    """Check SciPy candidates, distance/height/prominence, plateaus and input timing."""
    for fs in (100.0, 200.0):
        time = np.arange(round(120 * fs)) / fs
        x = np.cos(2 * np.pi * time / 4)
        original = x.copy()
        belt = detect_belt(x, fs)
        expected, properties = find_peaks(x, distance=round(1.5 * fs), prominence=0.5, height=-1.0)
        np.testing.assert_array_equal(belt["automatic"]["candidate_idx"], expected)
        np.testing.assert_array_equal(
            belt["automatic"]["candidate_prom"], properties["prominences"]
        )
        np.testing.assert_array_equal(belt["peak_idx"], np.arange(4 * fs, 120 * fs, 4 * fs))
        np.testing.assert_array_equal(belt["peak_t"], belt["peak_idx"] / fs)
        np.testing.assert_array_equal(belt["trough_t"], belt["trough_idx"] / fs)
        np.testing.assert_array_equal(belt["detection_waveform"], original)
        assert belt["smoothing_applied_samples"] == 0 and belt["rr_mean_bpm"] == 15.0
        assert not belt["source_waveform"].flags.writeable
        np.testing.assert_array_equal(x, original)
        validate_belt(belt, fs)
    plateau = detect_peaks(
        np.array([0.0, 1.0, 1.0, 1.0, 0.0]),
        1.0,
        DetectorSettings(min_peak_dist_sec=1.0, qc_enabled=False),
    )
    assert plateau["candidate_idx"].tolist() == [2]  # SciPy midpoint; MATLAB differs.
    rejected = detect_belt(
        np.cos(np.arange(1000) / 20), 100.0, DetectorSettings(min_peak_height=2.0)
    )
    assert not rejected["peak_idx"].size


def check_conservative_qc():
    """Check local split-peak removal reasons and the explicitly disabled QC path."""
    candidates = np.array([0, 800, 1600, 2000, 2400, 3200, 4000, 4800])
    prominence = np.ones(candidates.size)
    prominence[3] = 0.1
    kept, _, removed, reasons = peak_qc(candidates, prominence, 200.0, DetectorSettings())
    assert removed.tolist() == [2000] and reasons == ["short_ibi+rhythm+low_prominence"]
    np.testing.assert_array_equal(kept, candidates[candidates != 2000])
    result = peak_qc(candidates, prominence, 200.0, DetectorSettings(qc_enabled=False))
    np.testing.assert_array_equal(result[0], candidates)
    assert not result[2].size


def check_troughs_amplitudes_rates():
    """Check strict interior minima/percentiles, amplitude boundaries and MATLAB RR."""
    x = np.array([-9.0, 4.0, 1.0, 3.0, 2.0, -8.0, 5.0, 2.0, -3.0, 0.0, 4.0])
    belt = detect_belt(x, 1.0)
    belt["peak_idx"] = np.array([0, 5, 10])
    recompute(belt, 1.0)
    np.testing.assert_array_equal(belt["trough_idx"], [2, 8])
    np.testing.assert_allclose(belt["amp_exp"], [-10, -5, np.nan], equal_nan=True)
    np.testing.assert_allclose(belt["amp_insp"], [np.nan, -9, 7], equal_nan=True)
    np.testing.assert_allclose(belt["amp_sym"], [np.nan, -7, np.nan], equal_nan=True)
    assert belt["rr_mean_bpm"] == 12.0 and belt["rr_std_bpm"] == 0.0
    belt["peak_idx"] = np.array([0, 2, 10])
    recompute(belt, 1.0)
    assert belt["rr_mean_bpm"] == 60 / np.mean([2.0, 8.0])
    np.testing.assert_allclose(belt["rr_std_bpm"], np.std(60 / np.array([2.0, 8.0]), ddof=1))
    belt["peak_idx"] = np.array([0, 5, 10])
    recompute(belt, 1.0)
    for method, field in (
        ("inspiratory", "amp_insp"),
        ("expiratory", "amp_exp"),
        ("symmetric", "amp_sym"),
    ):
        belt["settings"]["amp_method"] = method
        recompute(belt, 1.0)
        np.testing.assert_array_equal(belt["amp"], belt[field])
    belt["settings"].update(trough_method="prctile", trough_prct=50.0)
    recompute(belt, 1.0)
    assert belt["trough_idx"][0] == 3 and belt["trough_val"][0] == x[3]
    belt["peak_idx"] = np.array([0, 1])
    recompute(belt, 1.0)
    assert np.isnan(belt["trough_idx"][0]) and not belt["ibi_valid"][0]
    assert "missing_trough" in belt["ibi_reason"][0]


def check_landmark_edits():
    """Check peak add/remove, pair-anchored moves/reset, invalid edits and stale fields."""
    fs = 200.0
    belt = detect_belt(np.cos(2 * np.pi * np.arange(24000) / 800), fs)
    left, right = map(int, belt["peak_idx"][:2])
    original_trough = belt["trough_idx"][0]
    move_trough(belt, fs, left, right, (left + 1) / fs)
    assert belt["trough_idx"][0] == left + 1
    distant = int(belt["peak_idx"][-2])
    edit_peak(belt, fs, 0, remove=distant)
    assert belt["trough_idx"][0] == left + 1
    edit_peak(belt, fs, distant / fs)
    assert distant in belt["peak_idx"]
    with np.testing.assert_raises(ValueError):
        edit_peak(belt, fs, distant / fs)
    before = belt["trough_idx"].copy()
    with np.testing.assert_raises(ValueError):
        move_trough(belt, fs, left, right, left / fs)
    np.testing.assert_array_equal(belt["trough_idx"], before)
    move_trough(belt, fs, left, right, reset=True)
    assert belt["trough_idx"][0] == original_trough
    move_trough(belt, fs, left, right, (left + 1) / fs)
    edit_peak(belt, fs, 0, remove=left)
    assert belt["trough_overrides"].shape == (0, 3)
    belt["settings"]["trough_prct"] = 10.0
    with np.testing.assert_raises(ValueError):
        validate_belt(belt, fs)
    assert belt["derived_stale"]
    recompute(belt, fs)
    assert not belt["derived_stale"]


def check_masks_and_missing_data():
    """Keep finite landmarks across QC flags; invalidate supported IBI over gaps/artifacts."""
    fs = 200.0
    x = np.cos(2 * np.pi * np.arange(24000) / 800)
    baseline = detect_belt(x, fs)
    artifact = np.zeros(len(x), bool)
    artifact[1100:1200] = True
    marked = detect_belt(x, fs, artifact=artifact)
    np.testing.assert_array_equal(marked["peak_idx"], baseline["peak_idx"])
    np.testing.assert_array_equal(marked["trough_idx"], baseline["trough_idx"])
    assert not marked["ibi_valid"][0] and np.isnan(marked["ibi"][0])
    assert "confirmed_artifact" in marked["ibi_reason"][0]
    unknown = detect_belt(x, fs, valid=np.zeros(len(x), bool))
    np.testing.assert_array_equal(unknown["trough_idx"], baseline["trough_idx"])
    assert not unknown["ibi_valid"].any()
    unavailable = detect_belt(x, fs, channel_status="unavailable")
    assert not unavailable["breath_valid"].any()
    x[1100:1110] = np.nan
    gap = detect_belt(x, fs)
    assert any("nonfinite" in reason for reason in gap["ibi_reason"])
    for values in (np.array([]), np.full(1000, np.nan), np.zeros(1000)):
        assert not detect_belt(values, fs)["peak_idx"].size
    for values, rate in ((np.zeros((10, 2)), fs), (np.zeros(10), 0)):
        with np.testing.assert_raises(ValueError):
            detect_belt(values, rate)
    with np.testing.assert_raises(ValueError):
        detect_belt(np.ones(10), fs, artifact=np.zeros(11, bool))
    for kwargs in (
        {"trough_method": "unknown"},
        {"min_peak_dist_sec": 0},
        {"qc_iterations": 6},
        {"trough_prct": 101},
    ):
        with np.testing.assert_raises(ValueError):
            DetectorSettings(**kwargs)


def check_preprocessed_adapter():
    """Check processed-rate timing, in-memory/loaded parity and a missing belt."""
    fs = 100.0
    time = np.arange(6000) / fs
    waveform = np.cos(2 * np.pi * time / 4)
    values = np.column_stack([waveform, waveform])
    identity = {
        "recording_id": "Sub999_M1",
        "subject": 999,
        "measurement": 1,
        "source_sha256": "0" * 64,
        "n_samples": 12000,
        "fs_hz": 200.0,
        "channels": list(BELTS),
    }
    qc = {"channel_ratings": dict.fromkeys(BELTS, "usable")}
    masks = {
        "valid": np.ones(values.shape, bool),
        "artifact": np.zeros(values.shape, bool),
        "interpolated": np.zeros(values.shape, bool),
    }
    processing = {"algorithm_version": "synthetic_preprocessing", "smoothing": {"passes": 1}}
    data = {
        "respiratory_analysis_source": "signals/smoothed",
        "processed_fs_hz": fs,
        "processed_time_sec": time,
        "channels": list(BELTS),
        "masks": masks,
        "signals": {"smoothed": values, "original": np.zeros((12000, 2))},
        "original_fs_hz": 200.0,
        "metadata": {"identity": identity, "qc": qc, "processing": processing},
    }
    automatic = detect_preprocessed(data)
    assert automatic["fs_hz"] == 100.0 and automatic["source"] == "signals/smoothed"
    assert automatic["belts"][BELTS[0]]["rr_mean_bpm"] == 15.0
    np.testing.assert_array_equal(automatic["belts"][BELTS[0]]["detection_waveform"], waveform)
    memory = SimpleNamespace(
        respiratory_signal=values,
        processed_fs_hz=fs,
        processed_time_sec=time,
        masks=masks,
        qc=qc,
        provenance=processing,
        recording=SimpleNamespace(samples=data["signals"]["original"], **identity),
    )
    from_memory = detect_preprocessed(memory)
    assert from_memory["identity"] == automatic["identity"]
    np.testing.assert_array_equal(
        from_memory["belts"][BELTS[0]]["peak_idx"], automatic["belts"][BELTS[0]]["peak_idx"]
    )
    one = deepcopy(data)
    one["channels"] = [BELTS[0]]
    one["signals"]["smoothed"] = values[:, :1]
    one["masks"] = {name: mask[:, :1] for name, mask in masks.items()}
    assert list(detect_preprocessed(one)["belts"]) == [BELTS[0]]
    invalid = deepcopy(data)
    invalid["processed_fs_hz"] = 200.0
    with np.testing.assert_raises(ValueError):
        detect_preprocessed(invalid)


def check_peak_detection_functions():
    """Run deterministic numerical, edit, support and stage-integration checks."""
    checks = (
        check_automatic_detection,
        check_conservative_qc,
        check_troughs_amplitudes_rates,
        check_landmark_edits,
        check_masks_and_missing_data,
        check_preprocessed_adapter,
    )
    for check in checks:
        logger.info("Checking %s...", check.__name__.removeprefix("check_"))
        check()
        logger.info("  PASS")
    return [check.__name__ for check in checks]


def check_saved_recording(path):
    """Read one real DBAT preprocessing HDF5 and check every belt without writing."""
    checksum = source_sha256(path)
    data = load_preprocessing_result(path)
    result = detect_preprocessed(data)
    for name, belt in result["belts"].items():
        validate_belt(belt, result["fs_hz"])
        column = data["channels"].index(name)
        waveform = data["signals"]["smoothed"][:, column]
        np.testing.assert_array_equal(belt["detection_waveform"], waveform)
        assert belt["smoothing_applied_samples"] == 0
        for left, right, trough in zip(
            belt["peak_idx"][:-1], belt["peak_idx"][1:], belt["trough_idx"], strict=True
        ):
            if right - left > 1 and np.isfinite(waveform[left : right + 1]).all():
                assert trough == left + 1 + np.argmin(waveform[left + 1 : right])
        logger.info(
            "%s: %d candidates, %d retained peaks, %d trough slots, %d assessable IBIs",
            name,
            len(belt["automatic"]["candidate_idx"]),
            len(belt["peak_idx"]),
            len(belt["trough_idx"]),
            belt["ibi_valid"].sum(),
        )
    assert checksum == source_sha256(path)
    logger.info("Real-recording peak checks passed; source SHA256 unchanged.")
    return result


def main(h5_path=None, *, verbosity=1):
    """Run synthetic checks and optional read-only validation of one real recording."""
    configure_logging(verbosity)
    check_peak_detection_functions()
    if h5_path is not None:
        return check_saved_recording(h5_path)
    logger.info("All peak checks passed. Add --h5 to check a saved DBAT recording.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5", type=Path, help="One saved DBAT preprocessing HDF5")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args()
    main(args.h5, verbosity=args.verbosity)
