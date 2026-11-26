"""Native-grid technical QC annotations, independent of the GUI and conditioning."""

import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from ..config import BELTS

QC_SCHEMA = "dbat_qc_v2"
READABLE_QC_SCHEMAS = {"dbat_qc_v1", QC_SCHEMA}
RATINGS = ("unassessed", "usable", "weak", "unavailable")


def continuous_runs(mask):
    """Yield half-open start/stop pairs for true runs in a 1-D boolean mask."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2], strict=True)


def utc_now():
    """Return an explicit UTC ISO timestamp for annotation provenance."""
    return datetime.now(timezone.utc).isoformat()


def recording_identity(recording):
    """Return the identity needed to reject QC from a different source/grid."""
    return {
        "recording_id": recording.recording_id,
        "subject": recording.subject,
        "measurement": recording.measurement,
        "source_sha256": recording.source_sha256,
        "n_samples": recording.samples.shape[0],
        "fs_hz": recording.fs_hz,
        "channels": list(recording.channels),
    }


def new_qc_annotations(recording, reviewer=""):
    """Create an unreviewed record; an empty artifact list is not clean evidence."""
    return {
        "schema_version": QC_SCHEMA,
        "identity": recording_identity(recording),
        "source": "dbat_manual_qc",
        "reviewer": reviewer,
        "created_at_utc": utc_now(),
        "accepted_at_utc": None,
        "review_status": "not_reviewed",
        "parent_qc_file": None,
        "channel_ratings": dict.fromkeys(recording.channels, "unassessed"),
        "rating_provenance": {},
        "artifacts": [],
        "candidate_resolutions": [],
        "reviewed_intervals": [],
        "edit_history": [],
    }


def make_interval(start, end, channels, fs_hz, reviewer):
    """Describe one zero-based half-open [start, end) annotation in samples/s."""
    return {
        "id": uuid4().hex,
        "schema_version": QC_SCHEMA,
        "start_sample": int(start),
        "end_sample": int(end),
        "start_sec": start / fs_hz,
        "end_sec": end / fs_hz,
        "channels": list(channels),
        "reviewer": reviewer,
        "created_at_utc": utc_now(),
        "source": "dbat_manual_qc",
    }


def merge_artifact_intervals(qc, fs_hz, reviewer):
    """Merge overlapping artifacts in place without changing per-channel coverage.

    Parameters
    ----------
    qc : dict
        Private GUI annotations. Replaced originals are retained in edit_history.
    fs_hz : float
        Native sampling frequency in Hz, used for the merged sample/second bounds.
    reviewer : str
        Reviewer responsible for the edit or normalization of the private copy.

    Notes
    -----
    Touching half-open intervals stay separate. Multi-channel annotations stay
    grouped only when the same original intervals contribute on those channels;
    independent channel annotations never acquire coupled deletion behaviour.
    """
    originals = qc["artifacts"]
    grouped = {}
    channels = dict.fromkeys(name for interval in originals for name in interval["channels"])
    for name in channels:
        runs = []
        for interval in sorted(
            (item for item in originals if name in item["channels"]),
            key=lambda item: (item["start_sample"], item["end_sample"]),
        ):
            start, end = interval["start_sample"], interval["end_sample"]
            if runs and start < runs[-1][1]:
                runs[-1][1] = max(runs[-1][1], end)
                runs[-1][2].append(interval)
            else:
                runs.append([start, end, [interval]])
        for start, end, contributors in runs:
            key = (start, end, tuple(sorted(item["id"] for item in contributors)))
            if key not in grouped:
                grouped[key] = {"channels": [], "originals": contributors}
            grouped[key]["channels"].append(name)
    merged = []
    for (start, end, _), group in grouped.items():
        source = group["originals"]
        if len(source) == 1 and set(group["channels"]) == set(source[0]["channels"]):
            merged.append(source[0])
        else:
            merged.append(make_interval(start, end, group["channels"], fs_hz, reviewer))
    kept = {item["id"] for item in originals} & {item["id"] for item in merged}
    replaced = [item for item in originals if item["id"] not in kept]
    if replaced:
        qc["artifacts"] = sorted(merged, key=lambda item: (item["start_sample"], item["end_sample"]))
        qc["edit_history"].append({
            "action": "merge_artifacts", "original_intervals": replaced,
            "merged_intervals": [item for item in merged if item["id"] not in kept],
            "reviewer": reviewer, "created_at_utc": utc_now(),
        })


def review_status(qc):
    """Describe accepted review evidence, including partial clean inspection."""
    if not qc["accepted_at_utc"]:
        return "not_reviewed"
    if qc["artifacts"]:
        return "reviewed_with_artifacts"
    return "reviewed_clean" if qc["reviewed_intervals"] else "not_reviewed"


def validate_qc_annotations(qc, recording, *, allow_unaccepted=False):
    """Validate native identity, interval coordinates, ratings and provenance.

    No intervals are clipped, silently repaired, or promoted to clean evidence.
    Channel rating and inspected coverage remain independent decisions.
    """
    if (qc.get("schema_version") not in READABLE_QC_SCHEMAS
            or qc.get("identity") != recording_identity(recording)):
        raise ValueError(
            "QC schema or source identity/hash/fs/channel grid does not match this recording"
        )
    ratings = qc.get("channel_ratings", {})
    if set(ratings) != set(recording.channels) or any(
        value not in RATINGS for value in ratings.values()
    ):
        raise ValueError("QC must contain one supported rating per input channel")
    has_evidence = any(value != "unassessed" for value in ratings.values())
    for name, rating in ratings.items():
        if rating != "unassessed":
            provenance = qc.get("rating_provenance", {}).get(name, {})
            if not provenance.get("reviewer", "").strip() or not provenance.get("created_at_utc"):
                raise ValueError(f"QC rating for {name} lacks reviewer/time provenance")
    for resolution in qc.get("candidate_resolutions", []):
        if (
            resolution.get("decision") not in {"dismiss", "artifact"}
            or not resolution.get("candidate_id")
            or len(resolution.get("preparation_signature", "")) != 64
            or not resolution.get("reasons")
            or (qc["schema_version"] == QC_SCHEMA and (
                resolution.get("source_stage") not in {"early_cleaning", "signal_quality"}
                or not isinstance(resolution.get("candidate"), dict)
                or resolution["candidate"].get("id") != resolution.get("candidate_id")
            ))
        ):
            raise ValueError(
                "Candidate resolution needs a decision and original detection provenance"
            )
    identifiers = set()
    for field in ("artifacts", "reviewed_intervals", "candidate_resolutions"):
        for interval in qc.get(field, []):
            start, end = interval["start_sample"], interval["end_sample"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= recording.samples.shape[0]
            ):
                raise ValueError(
                    f"QC {field} requires integer [start, end) bounds inside the recording"
                )
            channels = interval["channels"]
            if (
                not channels
                or len(set(channels)) != len(channels)
                or not set(channels) <= set(recording.channels)
            ):
                raise ValueError(f"QC {field} refers to missing or duplicate channels")
            for key, expected in (
                ("start_sec", start / recording.fs_hz),
                ("end_sec", end / recording.fs_hz),
            ):
                if not math.isclose(interval[key], expected, rel_tol=0, abs_tol=1e-9):
                    raise ValueError("QC seconds disagree with its native sample coordinates")
            if (interval.get("schema_version") not in READABLE_QC_SCHEMAS
                    or not interval.get("source")):
                raise ValueError("QC interval lacks a supported schema/source")
            if not interval.get("reviewer", "").strip() or not interval.get("created_at_utc"):
                raise ValueError("QC interval lacks reviewer/time provenance")
            if not interval.get("id") or interval["id"] in identifiers:
                raise ValueError("QC interval IDs must be nonempty and unique")
            identifiers.add(interval["id"])
            has_evidence = True
    baseline = qc.get("baseline_correction", {})
    if (not isinstance(baseline, dict) or not set(baseline) <= set(BELTS)
            or any(type(value) is not bool for value in baseline.values())):
        raise ValueError("QC baseline_correction must map respiratory belt names to booleans")
    decision = qc.get("polarity")
    if decision is not None:
        if "choice" not in decision or decision["choice"] not in (
            None,
            "Resp-Lungs",
            "Resp-Diaphragm",
        ):
            raise ValueError("Invalid manual belt-polarity choice")
        if (
            not decision.get("reviewer", "").strip()
            or not decision.get("created_at_utc")
            or decision.get("source") != "dbat_manual_qc"
            or not isinstance(decision.get("preparation_signature"), str)
            or len(decision["preparation_signature"]) != 64
            or type(decision.get("acknowledged_warning")) is not bool
        ):
            raise ValueError("Polarity decision lacks reviewer/time/source/cleaning provenance")
        assessment = decision.get("assessment", {})
        if (
            not allow_unaccepted
            and assessment.get("polarity_warning")
            and decision["choice"] is None
            and not decision["acknowledged_warning"]
        ):
            raise ValueError(
                "Keeping None under a polarity warning requires explicit acknowledgement"
            )
        has_evidence = True
    if has_evidence and (
        (not qc.get("accepted_at_utc") and not allow_unaccepted)
        or not qc.get("reviewer", "").strip()
    ):
        raise ValueError("Manual QC evidence requires an accepted review and a reviewer")
    if qc.get("review_status") != review_status(qc):
        raise ValueError("QC review_status disagrees with its accepted interval evidence")


def annotation_masks(qc, recording, *, allow_unaccepted=False):
    """Union original intervals into independent (samples, channels) boolean masks.

    Original overlapping/touching annotations and their provenance stay intact.
    Confirmed artifacts count as inspected evidence but never as clean support.
    """
    validate_qc_annotations(qc, recording, allow_unaccepted=allow_unaccepted)
    masks = {key: np.zeros(recording.samples.shape, dtype=bool) for key in ("artifact", "reviewed")}
    for field, key in (("reviewed_intervals", "reviewed"), ("artifacts", "artifact")):
        for interval in qc[field]:
            for name in interval["channels"]:
                column = recording.channels.index(name)
                start, end = interval["start_sample"], interval["end_sample"]
                masks[key][start:end, column] = True
                if key == "artifact":
                    masks["reviewed"][start:end, column] = True
    return masks


def active_candidate_resolutions(qc, prepared, candidates=None):
    """Return decisions matching exact current candidate identity and native bounds."""
    candidates = {
        item["id"]: item
        for item in (prepared["candidates"] if candidates is None else candidates)
    }
    active = {}
    for decision in qc.get("candidate_resolutions", []):
        candidate = candidates.get(decision["candidate_id"])
        if (
            candidate is not None
            and decision["preparation_signature"] == candidate["preparation_signature"]
            and decision.get("source_stage", "early_cleaning") == candidate["source_stage"]
            and decision["channels"] == [candidate["channel"]]
            and decision["start_sample"] == candidate["start_sample"]
            and decision["end_sample"] == candidate["end_sample"]
        ):
            if decision["decision"] == "artifact" and not any(
                decision["channels"][0] in interval["channels"]
                and interval["start_sample"] <= decision["start_sample"]
                and interval["end_sample"] >= decision["end_sample"]
                for interval in qc["artifacts"]
            ):
                continue  # Removing the confirmed artifact reopens its warning.
            active[candidate["id"]] = decision
    return active


def review_candidates(qc, prepared, quality):
    """Combine current stage warnings and retained reviewer evidence snapshots."""
    candidates = {
        item["id"]: item for item in prepared["candidates"] + quality["candidates"]
    }
    signatures = {
        "early_cleaning": prepared["signature"],
        "signal_quality": quality.get("signature"),
    }
    for resolution in qc.get("candidate_resolutions", []):
        candidate = resolution.get("candidate")
        if (
            candidate is not None
            and candidate["preparation_signature"]
            == signatures[candidate["source_stage"]]
        ):
            candidates.setdefault(candidate["id"], candidate)
    return sorted(candidates.values(), key=lambda item: (item["start_sample"], item["channel"], item["source_stage"]))


def unresolved_candidate_mask(
    qc, prepared, candidates=None, *, source_stage="early_cleaning"
):
    """Build native unresolved warnings without changing evidence or review coverage."""
    candidates = prepared["candidates"] if candidates is None else candidates
    active = active_candidate_resolutions(qc, prepared, candidates)
    unresolved = np.zeros(prepared["corrected"].shape, dtype=bool)
    for candidate in candidates:
        if candidate["source_stage"] == source_stage and candidate["id"] not in active:
            column = prepared["identity"]["channels"].index(candidate["channel"])
            unresolved[candidate["start_sample"] : candidate["end_sample"], column] = (
                True
            )
    return unresolved


def add_analysis_masks(masks, corrected, recording, qc, selected_channels, automatic_masks):
    """Attach numerical availability and existing QC eligibility to native masks.

    Updates masks in place. Corrected samples have shape (native_samples,
    channels); quality selection, coverage, ratings and artifacts remain
    independent of numerical conditioning.
    """
    masks.update(automatic_masks)
    masks["finite"] = np.isfinite(corrected)
    usable = np.array([qc["channel_ratings"][name] == "usable" for name in recording.channels])
    unknown = np.array(
        [qc["channel_ratings"][name] in {"weak", "unassessed"} for name in recording.channels]
    )
    selected = np.array([name in selected_channels for name in recording.channels])
    masks["quality_selected"] = np.broadcast_to(selected, corrected.shape).copy()
    masks["uncertain"] = ~masks["reviewed"] | unknown[None, :] | ~masks["quality_selected"]
    masks["eligible"] = (
        masks["quality_selected"]
        & masks["finite"]
        & masks["reviewed"]
        & ~masks["artifact"]
        & usable[None, :]
    )
