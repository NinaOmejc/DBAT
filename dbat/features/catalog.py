"""One registry for Stage 5b feature selection, units and dependencies.

Catch22 names adapted from Nina Omejc's MIT-licensed MAGMA-ml src/features/catalog.py:
https://github.com/NinaOmejc/magma-ml/blob/815b01a7a52baac087d504308fa6d633e98ab984/src/features/catalog.py
Copyright (c) 2026 Nina Omejc. See LICENSE.
"""

from ..config import BELTS

SUPPORTED_CHANNELS = (*BELTS, "SpO2")
CATCH22_NAMES = (
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
)
# name: (numerical group, applicable channels, unit, direct dependencies)
REGISTRY = {}
for group, names, units, dependencies in (
    (
        "breaths",
        (
            "ibi",
            "respiratory_rate",
            "amplitude",
            "amp_ratio",
            "global_amp_ratio",
            "inspiratory_duration",
            "expiratory_duration",
            "ie_ratio",
        ),
        ("s", "breaths/min", "source units", "1", "1", "s", "s", "1"),
        ("landmarks",),
    ),
    (
        "local",
        (
            "local_respiratory_rate",
            "ibi_cov",
            "ibi_robust_cov",
            "waveform_excursion",
            "excursion_ratio",
        ),
        ("breaths/min", "1", "1", "source units", "1"),
        ("waveform", "landmarks"),
    ),
    (
        "envelope",
        ("envelope", "envelope_cov"),
        ("normalized source units", "1"),
        ("waveform",),
    ),
    ("eami", ("eami",), ("1",), ("waveform",)),
    (
        "spectral",
        (
            "dominant_frequency",
            "spectral_bandwidth",
            "spectral_entropy",
            "dominant_power_fraction",
            "second_harmonic_power_fraction",
        ),
        ("Hz", "Hz", "1", "1", "1"),
        ("waveform",),
    ),
):
    for name, unit in zip(names, units, strict=True):
        REGISTRY[name] = {
            "group": group,
            "channels": BELTS,
            "unit": unit,
            "dependencies": dependencies,
        }
for group, names in (
    (
        "balance",
        (
            "thoracic_amplitude_median",
            "abdominal_amplitude_median",
            "ta_ratio",
            "ta_log_ratio",
            "thoracic_fraction",
        ),
    ),
    ("phase", ("phase_difference", "absolute_phase_difference", "phase_consistency")),
    ("coherence", ("spectral_coherence",)),
):
    for name in names:
        REGISTRY[name] = {
            "group": group,
            "channels": ("two_belt",),
            "unit": "degrees" if "difference" in name else "1",
            "dependencies": ("Resp-Lungs", "Resp-Diaphragm"),
        }
for name, unit in (("oxygen_saturation", "%"), ("spo2_decrease", "percentage points")):
    REGISTRY[name] = {
        "group": "spo2",
        "channels": ("SpO2",),
        "unit": unit,
        "dependencies": ("waveform",),
    }
for name in CATCH22_NAMES:
    REGISTRY[name] = {
        "group": "catch22",
        "channels": SUPPORTED_CHANNELS,
        "unit": "Catch22 native units",
        "dependencies": ("waveform",),
    }
for name, dependency in (
    ("amp_ratio", "session_reference"),
    ("global_amp_ratio", "global_reference"),
    ("excursion_ratio", "waveform_reference"),
    ("spo2_decrease", "spo2_reference"),
    ("second_harmonic_power_fraction", "landmarks"),
):
    REGISTRY[name]["dependencies"] += (dependency,)
for name in (
    "thoracic_amplitude_median",
    "abdominal_amplitude_median",
    "ta_ratio",
    "ta_log_ratio",
    "thoracic_fraction",
):
    REGISTRY[name]["dependencies"] += ("session_references",)


def select_features(names):
    """Expand family selectors and reject unknown or repeated names."""
    if names is None or names == []:
        return list(REGISTRY)
    if not isinstance(names, (list, tuple)) or any(
        not isinstance(n, str) for n in names
    ):
        raise ValueError("features must be a list of feature or family names")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate feature selection")
    families = {
        "respiratory": [n for n, m in REGISTRY.items() if m["channels"] == BELTS],
        "thoracoabdominal": [
            n for n, m in REGISTRY.items() if m["channels"] == ("two_belt",)
        ],
        "spo2": ["oxygen_saturation", "spo2_decrease"],
        "catch22": list(CATCH22_NAMES),
    }
    result = []
    for name in names:
        if name not in REGISTRY and name not in families:
            raise ValueError(f"Unknown feature/family: {name}")
        for item in families.get(name, [name]):
            if item not in result:
                result.append(item)
    return result
