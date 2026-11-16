#!/usr/bin/env python3
"""Validate local Waymo shards and write a scenario eligibility/split CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nexgen.splits import (  # noqa: E402
    DEFAULT_SPLIT_SEED,
    TARGET_NODE_COUNTS,
    assign_scenario_splits,
    any_valid_vehicle_ids,
    concurrent_vehicle_counts,
    fully_valid_vehicle_ids,
)
from nexgen.waymo import Scenario, read_scenarios  # noqa: E402


def _find_shard(data_dir: Path, shard_index: int) -> Path:
    suffix = f"training_20s.tfrecord-{shard_index:05d}-of-01000"
    matches = sorted(data_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"missing Waymo shard {shard_index:05d} in {data_dir}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple files match shard {shard_index:05d}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def _selection_row(
    shard_path: Path, scenario: Scenario, split: str
) -> dict[str, object]:
    full_episode_ids = fully_valid_vehicle_ids(scenario)
    active_counts = concurrent_vehicle_counts(scenario)
    row: dict[str, object] = {
        "shard": shard_path.name,
        "scenario_id": scenario.scenario_id,
        "split": split,
        "raw_timestamp_count": scenario.raw_timestamp_count,
        "track_state_count_min": scenario.track_state_count_min,
        "track_state_count_max": scenario.track_state_count_max,
        "aligned_frame_count": scenario.aligned_frame_count,
        "frame_count_mismatch": scenario.frame_count_mismatch,
        "truncated_timestamp_frames": (
            scenario.raw_timestamp_count - scenario.aligned_frame_count
        ),
        "truncated_track_frames_max": (
            scenario.track_state_count_max - scenario.aligned_frame_count
        ),
        "duration_seconds": (
            scenario.timestamps[-1] - scenario.timestamps[0]
        ),
        "total_track_count": scenario.total_track_count,
        "vehicle_track_count": len(scenario.vehicles),
        "vehicle_tracks_any_valid": len(any_valid_vehicle_ids(scenario)),
        "vehicle_tracks_valid_all_frames": len(full_episode_ids),
        "minimum_concurrent_vehicles": min(active_counts, default=0),
        "peak_concurrent_vehicles": max(active_counts, default=0),
    }
    for target_count in TARGET_NODE_COUNTS:
        row[f"eligible_{target_count}"] = (
            len(full_episode_ids) >= target_count
        )
    return row


def _summary(rows: list[dict[str, object]], first: int, last: int) -> dict[str, object]:
    split_counts = Counter(str(row["split"]) for row in rows)
    eligible_by_split: dict[str, dict[str, int]] = {}
    for target_count in TARGET_NODE_COUNTS:
        scale_counts = Counter(
            str(row["split"])
            for row in rows
            if row[f"eligible_{target_count}"] is True
        )
        eligible_by_split[str(target_count)] = {
            "train": scale_counts["train"],
            "development": scale_counts["development"],
            "held_out": scale_counts["held_out"],
            "total": sum(scale_counts.values()),
        }
    return {
        "shard_range": f"{first:05d}-{last:05d}",
        "scenario_count": len(rows),
        "unique_scenario_ids": len({row["scenario_id"] for row in rows}),
        "frame_count_mismatch_scenarios": sum(
            row["frame_count_mismatch"] is True for row in rows
        ),
        "scenarios_by_split": {
            split: split_counts[split]
            for split in ("train", "development", "held_out")
        },
        "eligible_scenarios_by_target_size": eligible_by_split,
    }


def prepare_selection(
    data_dir: Path,
    output_dir: Path,
    first_shard: int,
    last_shard: int,
    split_seed: int,
    overwrite: bool = False,
) -> tuple[Path, dict[str, object]]:
    if first_shard < 0 or last_shard < first_shard or last_shard > 999:
        raise ValueError("shard range must satisfy 0 <= first <= last <= 999")

    shard_paths = [
        _find_shard(data_dir, index)
        for index in range(first_shard, last_shard + 1)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = (
        output_dir
        / f"scenario_selection_{first_shard:05d}-{last_shard:05d}.csv"
    )
    if selection_path.exists() and not overwrite:
        raise FileExistsError(
            f"{selection_path} already exists; pass --overwrite to replace it"
        )

    loaded_scenarios = list(read_scenarios(shard_paths))
    if not loaded_scenarios:
        raise ValueError("no scenarios found in the selected Waymo shards")
    assignments = assign_scenario_splits(
        (scenario for _, scenario in loaded_scenarios),
        seed=split_seed,
    )
    rows = [
        _selection_row(
            shard_path,
            scenario,
            assignments[scenario.scenario_id],
        )
        for shard_path, scenario in loaded_scenarios
    ]

    fieldnames = list(rows[0])
    with selection_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summary(rows, first_shard, last_shard)
    summary["split_seed"] = split_seed
    summary["selection_csv"] = str(selection_path)
    return selection_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "waymo" / "raw",
        help="folder containing downloaded uncompressed TFRecord shards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "waymo",
        help="local folder for the generated selection CSV",
    )
    parser.add_argument("--first-shard", type=int, default=0)
    parser.add_argument("--last-shard", type=int, default=9)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this shard range's existing selection CSV",
    )
    args = parser.parse_args()

    _, summary = prepare_selection(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        first_shard=args.first_shard,
        last_shard=args.last_shard,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
