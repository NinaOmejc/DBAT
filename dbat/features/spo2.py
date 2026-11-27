"""Current-QC saturation and signed decrease from the existing static reference.

Reference-minus-observed convention:
https://github.com/NinaOmejc/magma-labeling/blob/a7081d2e87131bb05563431778ad10b2865296a7/src/label_detection/detect_desaturation.m
Copyright (c) 2026 Nina Omejc. See LICENSE.
"""

import numpy as np

from .common import observations


def saturation_measurements(values, fs, reference, selected):
    """Retain percentages; positive decrease means observed SpO2 is below reference."""
    t = np.arange(len(values)) / fs
    out = {}
    if "oxygen_saturation" in selected:
        out["oxygen_saturation"] = observations(
            values, t, end=t + 1 / fs, resolution="sample"
        )
    if "spo2_decrease" in selected:
        baseline = reference.get("median_percent")
        baseline = (
            float(baseline)
            if reference.get("available")
            and baseline is not None
            and np.isfinite(baseline)
            else np.nan
        )
        out["spo2_decrease"] = observations(
            baseline - values,
            t,
            end=t + 1 / fs,
            resolution="sample",
            definition="static reference minus observed; not absolute value",
        )
    return out
