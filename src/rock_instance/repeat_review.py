"""Deterministic, isolated repeat-review support for corrected calibration evidence."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from src.rock_instance.annotations import (
    PROTOCOL_V2_CALIBRATION_RESOLVED,
    unresolved_candidate_component_ids,
    validate_review_state,
)


REPEAT_SELECTION_VERSION = "calibration-repeat-v1"
REPEAT_SELECTION_SEED = 42


def _rank(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"{REPEAT_SELECTION_VERSION}:{seed}:{image_id}".encode("utf-8")).hexdigest()


def select_repeat_image_ids(state: dict[str, Any], *, target_size: int, seed: int = REPEAT_SELECTION_SEED) -> list[str]:
    """Choose a state-independent subset only after corrected calibration is fully complete."""
    validate_review_state(state)
    if state.get("protocol", {}).get("version") != PROTOCOL_V2_CALIBRATION_RESOLVED:
        raise ValueError("Repeat selection requires a v2.0-calibration-resolved state.")
    scope = state.get("review_scope", {})
    if scope.get("name") != "calibration":
        raise ValueError("Repeat selection must start from the completed primary calibration scope.")
    image_ids = list(scope["image_ids"])
    if not 1 <= target_size <= len(image_ids):
        raise ValueError("Repeat target_size must be between one and the completed calibration scope size.")
    incomplete = [
        image_id
        for image_id in image_ids
        if state["images"][image_id]["review_status"] != "reviewed"
        or unresolved_candidate_component_ids(state, image_id)
    ]
    if incomplete:
        raise ValueError(f"Repeat selection is blocked until every corrected calibration image is final: {incomplete}")
    return sorted(image_ids, key=lambda image_id: (_rank(seed, image_id), image_id))[:target_size]


def initialize_isolated_repeat_state(state: dict[str, Any], repeat_image_ids: list[str]) -> dict[str, Any]:
    """Create an empty, independent repeat-review state without changing primary evidence."""
    validate_review_state(state)
    primary_scope = state.get("review_scope", {})
    if primary_scope.get("name") != "calibration":
        raise ValueError("An isolated repeat state can only be initialized from primary calibration.")
    primary_image_ids = set(primary_scope["image_ids"])
    if not repeat_image_ids or len(set(repeat_image_ids)) != len(repeat_image_ids):
        raise ValueError("Repeat review requires unique selected image IDs.")
    if not set(repeat_image_ids) <= primary_image_ids:
        raise ValueError("Repeat review includes images outside the primary calibration scope.")
    repeat_state = copy.deepcopy(state)
    for image in repeat_state["images"].values():
        image["annotations"] = []
        image["review_status"] = "unreviewed"
        image["reviewer"] = None
        if "candidate_component_ids" in image:
            image["obvious_candidate_independent_rock_observed"] = False
            image["candidate_independent_observation_note"] = ""
    repeat_state["resolution_records"] = []
    repeat_state["review_scope"] = {
        **primary_scope,
        "name": "calibration_repeat",
        "image_ids": list(repeat_image_ids),
    }
    repeat_state["repeat_review"] = {
        "selection_version": REPEAT_SELECTION_VERSION,
        "selection_seed": REPEAT_SELECTION_SEED,
        "source_protocol_version": PROTOCOL_V2_CALIBRATION_RESOLVED,
        "primary_calibration_snapshot_sha256": hashlib.sha256(
            repr(state).encode("utf-8")
        ).hexdigest(),
    }
    validate_review_state(repeat_state)
    return repeat_state