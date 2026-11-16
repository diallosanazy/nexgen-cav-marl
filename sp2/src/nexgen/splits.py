"""Scenario eligibility, reproducible node sampling, and split assignment."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable
from typing import Literal

from .waymo import Scenario

SplitName = Literal["train", "development", "held_out"]
TARGET_NODE_COUNTS = (8, 20, 50)
DEFAULT_SPLIT_SEED = 20261110


def fully_valid_vehicle_ids(scenario: Scenario) -> tuple[int, ...]:
    """Return vehicle IDs valid at every aligned frame of a scenario."""
    return tuple(
        track.track_id
        for track in scenario.vehicles
        if len(track.states) == scenario.aligned_frame_count
        and all(state.has_position for state in track.states)
    )


def any_valid_vehicle_ids(scenario: Scenario) -> tuple[int, ...]:
    """Return vehicle IDs with at least one valid, positioned aligned state."""
    return tuple(
        track.track_id
        for track in scenario.vehicles
        if any(state.has_position for state in track.states)
    )


def concurrent_vehicle_counts(scenario: Scenario) -> tuple[int, ...]:
    """Count positioned, valid vehicle states at every aligned motion frame."""
    return tuple(
        sum(track.states[frame_index].has_position for track in scenario.vehicles)
        for frame_index in range(scenario.aligned_frame_count)
    )


def select_vehicle_ids(
    scenario: Scenario, target_count: int, seed: int
) -> tuple[int, ...]:
    """Select a deterministic, nested sample of full-episode vehicle tracks.

    Calling this with the same scenario and seed for target sizes 8, 20, and 50
    yields nested node sets, which keeps scale comparisons matched.
    """
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    eligible_ids = sorted(fully_valid_vehicle_ids(scenario))
    if len(eligible_ids) < target_count:
        raise ValueError(
            f"{scenario.scenario_id}: requested {target_count} full-episode "
            f"vehicles, found {len(eligible_ids)}"
        )

    selection_order = random.Random(seed).sample(eligible_ids, len(eligible_ids))
    return tuple(selection_order[:target_count])


def _stable_scenario_key(scenario_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{scenario_id}".encode("utf-8")).digest()


def _eligibility_stratum(scenario: Scenario) -> str:
    count = len(fully_valid_vehicle_ids(scenario))
    if count >= 50:
        return "50+"
    if count >= 20:
        return "20-49"
    if count >= 8:
        return "8-19"
    return "under-8"


def assign_scenario_splits(
    scenarios: Iterable[Scenario],
    seed: int = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.70,
    development_fraction: float = 0.15,
) -> dict[str, SplitName]:
    """Assign complete scenarios to reproducible, size-stratified partitions.

    Each eligibility stratum is ordered by a stable hash of its scenario IDs,
    then divided 70/15/15. This keeps high-node scenarios represented in each
    partition while preventing frames from a scenario crossing splits.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1")
    if train_fraction + development_fraction >= 1:
        raise ValueError("train and development fractions must sum to less than 1")

    scenarios_by_stratum: dict[str, list[Scenario]] = {}
    seen_ids: set[str] = set()
    for scenario in scenarios:
        if not scenario.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if scenario.scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario_id: {scenario.scenario_id}")
        seen_ids.add(scenario.scenario_id)
        stratum = _eligibility_stratum(scenario)
        scenarios_by_stratum.setdefault(stratum, []).append(scenario)

    assignments: dict[str, SplitName] = {}
    for stratum_scenarios in scenarios_by_stratum.values():
        ordered = sorted(
            stratum_scenarios,
            key=lambda scenario: (
                _stable_scenario_key(scenario.scenario_id, seed),
                scenario.scenario_id,
            ),
        )
        count = len(ordered)
        train_count = math.floor(count * train_fraction)
        development_count = math.floor(count * development_fraction)
        for index, scenario in enumerate(ordered):
            if index < train_count:
                split: SplitName = "train"
            elif index < train_count + development_count:
                split = "development"
            else:
                split = "held_out"
            assignments[scenario.scenario_id] = split
    return assignments
