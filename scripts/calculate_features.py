"""Calculate Stage 5b from compatible saved revisions; never open review GUIs.

Run: python scripts/calculate_features.py --config configs/default.toml
Or supply --preprocessing FILE --landmarks FILE --references FILE for one triplet.
"""

import argparse
import tomllib
from pathlib import Path

from dbat.config import _validate_selection, validate_feature_config
from dbat.features import calculate_features
from dbat.features.io import compatible_inputs
from dbat.logging import configure_logging, logger

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.toml"


def main(
    config_path=DEFAULT_CONFIG, *, preprocessing=None, landmarks=None, references=None
):
    """Process configured recordings using linked revisions, or one explicit triplet."""
    path = Path(config_path).resolve(strict=True)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    config["config_path"] = str(path)
    validate_feature_config(config["features"])
    _validate_selection(config["run"])
    configure_logging(config["run"]["verbosity"])
    supplied = (preprocessing, landmarks, references)
    if any(p is not None for p in supplied):
        if not all(p is not None for p in supplied):
            raise ValueError(
                "Supply all three saved paths: --preprocessing, --landmarks, --references"
            )
        triplets = [supplied]
    else:

        def resolve(value):
            value = Path(value)
            return value if value.is_absolute() else path.parent / value

        excluded = {tuple(pair) for pair in config["run"]["exclude_recordings"]}
        triplets = [
            compatible_inputs(
                f"Sub{s}_M{m}",
                resolve(config["paths"]["output_dir"]),
                resolve(config["peak_detection"]["output_dir"]),
                resolve(config["features"]["reference_dir"]),
            )
            for s in config["run"]["subjects"]
            for m in config["run"]["measurements"]
            if (s, m) not in excluded
        ]
    outputs = []
    for data, landmark_path, reference_path in triplets:
        result = calculate_features(
            data, landmarks=landmark_path, references=reference_path, config=config
        )
        outputs.append(result["output_path"])
        logger.info("Saved features: %s", result["output_path"])
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preprocessing", type=Path)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    main(
        args.config,
        preprocessing=args.preprocessing,
        landmarks=args.landmarks,
        references=args.references,
    )
