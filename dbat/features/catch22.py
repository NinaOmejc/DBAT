"""Gap-aware Catch22, adapted from MAGMA-ml without joining invalid gaps.

https://github.com/NinaOmejc/magma-ml/blob/815b01a7a52baac087d504308fa6d633e98ab984/src/features/catch22.py
Copyright (c) 2026 Nina Omejc. See LICENSE. Backend pinned to aeon 1.5.0;
its Catch22 definitions match 1.3.0 (AST comparison), without pycatch22.
"""

import importlib.metadata

import numpy as np

from ..preprocessing.annotations import continuous_runs
from .common import observations


def catch22_measurements(values, valid, fs, names, intervals, cfg):
    """Duration-weight each defined feature across >=60-s runs with >=80% coverage.

    Values must already have MAGMA-ml's channel normalization. Intervals are
    zero-based, half-open processed indices. Undefined values are never filled.
    """
    from aeon.transformations.collection.feature_based import Catch22

    if importlib.metadata.version("aeon") != "1.5.0":
        raise ValueError(
            "Reproducible Catch22 requires the approved aeon==1.5.0 backend"
        )
    transformer = Catch22(
        features=list(names),
        catch24=False,
        outlier_norm=True,
        replace_nans=False,
        use_pycatch22=False,
        n_jobs=1,
    )
    results, coverage, support, audit = [], [], [], []
    for interval in intervals:
        first, stop = interval["start_idx"], interval["end_idx"]
        runs = [
            (int(first + a), int(first + b))
            for a, b in continuous_runs(
                valid[first:stop] & np.isfinite(values[first:stop])
            )
            if (b - a) / fs >= cfg["min_segment_sec"]
        ]
        lengths = np.array([b - a for a, b in runs], float)
        fraction = lengths.sum() / (stop - first)
        row, weights = np.full(len(names), np.nan), np.zeros(len(names))
        details = []
        if fraction >= cfg["min_coverage"] and len(runs):
            vectors = []
            for a, b in runs:
                x = values[a:b]
                vector = (
                    np.full(len(names), np.nan)
                    if np.ptp(x) <= np.finfo(float).eps
                    else np.asarray(
                        transformer.fit_transform([x[None, :]]), float
                    ).reshape(-1)
                )
                vectors.append(vector)
                details.append(
                    {
                        "start_idx": a,
                        "end_idx": b,
                        "start_sec": a / fs,
                        "end_sec": b / fs,
                    }
                )
            vectors = np.asarray(vectors)
            for j in range(len(names)):
                keep = np.isfinite(vectors[:, j])
                if keep.any():
                    weights[j] = lengths[keep].sum() / fs
                    row[j] = np.average(vectors[keep, j], weights=lengths[keep])
        results.append(row)
        coverage.append(fraction)
        support.append(weights)
        audit.append(
            {
                "eligible_segments": [{"start_idx": a, "end_idx": b} for a, b in runs],
                "computed_segments": details,
            }
        )
    starts, ends = (
        np.array([i["start_sec"] for i in intervals]),
        np.array([i["end_sec"] for i in intervals]),
    )
    results, support = np.asarray(results), np.asarray(support)
    return {
        name: observations(
            results[:, j],
            (starts + ends) / 2,
            start=starts,
            end=ends,
            coverage=coverage,
            resolution="analysis_interval",
            valid_duration_sec=support[:, j],
            backend="aeon==1.5.0",
            aggregation="duration_weighted_valid_segments",
            minimum_segment_sec=cfg["min_segment_sec"],
            minimum_coverage=cfg["min_coverage"],
            segment_audit=audit,
        )
        for j, name in enumerate(names)
    }
