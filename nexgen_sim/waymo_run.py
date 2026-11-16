"""Unified real-Waymo runner. Run from the repository root after installing sp2."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

from nexgen.splits import select_vehicle_ids
from nexgen.waymo import read_scenarios

from .engine import Simulator, WAIT
from .mobility import ShortestPathReference, schedule_snapshots
from .traffic import Stream, poisson
from .waymo_adapter import waymo_snapshots


def run_scenario(scenario, *, cohort_size=8, selection_seed=42, radius=100,
                 transmission=1000, propagation=0, processing=0, wait_delay=100_000,
                 queue_capacity=10, traffic_seed=42, streams=None, rate=5,
                 lifetime=1_000_000, frozen=False):
    ids = select_vehicle_ids(scenario, cohort_size, selection_seed)
    snapshots = waymo_snapshots(scenario, ids, radius=radius,
                               transmission=transmission, propagation=propagation)
    horizon = snapshots[-1].at
    if streams is None:
        # Explicit, exported diagnostic pairs: a ring of sorted selected IDs.
        nodes = snapshots[0].topology.nodes
        streams = [Stream(f"vehicle-{source}", source, nodes[(i + 1) % len(nodes)],
                          rate, lifetime, 0, horizon) for i, source in enumerate(nodes)]
    if any(s.start > horizon or s.stop > horizon for s in streams):
        raise ValueError("traffic window exceeds available scenario duration")
    workload = poisson(streams, seed=traffic_seed, topology=snapshots[0].topology)
    sim = Simulator.from_topology(snapshots[0].topology, lambda _: WAIT,
                                  processing=processing, wait_delay=wait_delay,
                                  queue_capacity=queue_capacity)
    sim.policy = ShortestPathReference(lambda: sim.topology)
    schedule_snapshots(sim, snapshots[:1] if frozen else snapshots)
    workload.replay(sim)
    sim.run(horizon)
    updates = [e for e in sim.log if e["event"] == "topology_updated"]
    changes = sum(len(e["added"]) + len(e["removed"]) for e in updates)
    metrics = sim.metrics()
    summary = {k: v for k, v in metrics.items() if k not in ("nodes", "messages")}
    positions_changed = any(s.topology.node_attributes(n)["position"] !=
                            snapshots[0].topology.node_attributes(n)["position"]
                            for s in snapshots[1:] for n in s.topology.nodes)
    summary.update(scenario_id=scenario.scenario_id, horizon_us=horizon,
                   frame_count=len(snapshots), applied_snapshots=len(updates),
                   directed_link_changes=changes,
                   source_positions_changed=positions_changed,
                   positions_changed=positions_changed and not frozen,
                   transmission_attempts=sum(m.attempts for m in sim.messages.values()))
    config = dict(scenario_id=scenario.scenario_id, selected_track_ids=list(ids),
                  cohort_size=cohort_size, selection_seed=selection_seed, radius_m=radius,
                  transmission_us=transmission, propagation_us=propagation, processing_us=processing,
                  wait_delay_us=wait_delay, queue_capacity=queue_capacity, traffic_seed=traffic_seed,
                  horizon_us=horizon, frozen=frozen, streams=[asdict(s) for s in streams],
                  timestamp_origin_seconds=scenario.timestamps[0],
                  raw_timestamp_count=scenario.raw_timestamp_count,
                  aligned_frame_count=scenario.aligned_frame_count,
                  frame_count_mismatch=scenario.frame_count_mismatch,
                  policy="full_current_graph_shortest_path", python_version=platform.python_version())
    return sim, workload, snapshots, config, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrite a run")
    parser.add_argument("--cohort-size", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--radius", type=float, default=100)
    parser.add_argument("--transmission", type=int, default=1000, help="microseconds")
    parser.add_argument("--propagation", type=int, default=0, help="microseconds")
    parser.add_argument("--processing", type=int, default=0, help="microseconds")
    parser.add_argument("--wait-delay", type=int, default=100_000, help="microseconds")
    parser.add_argument("--queue-capacity", type=int, default=10)
    parser.add_argument("--traffic-seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=5, help="messages/second per default stream")
    parser.add_argument("--lifetime", type=int, default=1_000_000, help="microseconds")
    parser.add_argument("--streams", type=Path, help="JSON list of Stream field objects; overrides rate/lifetime")
    parser.add_argument("--frozen", action="store_true", help="diagnostic control: hold first topology")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; choose a new run directory")
    scenario = next((s for _, s in read_scenarios([args.shard]) if s.scenario_id == args.scenario_id), None)
    if scenario is None:
        parser.error("scenario ID not found in shard")
    streams = None if args.streams is None else [Stream(**s) for s in json.loads(args.streams.read_text())]
    options = vars(args).copy()
    for key in ("shard", "scenario_id", "output", "streams"):
        options.pop(key)
    sim, workload, snapshots, config, summary = run_scenario(scenario, streams=streams, **options)
    with args.shard.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    config["shard"] = dict(path=str(args.shard.resolve()), bytes=args.shard.stat().st_size, sha256=digest)
    args.output.mkdir(parents=True)
    for name, value in (("config", config), ("summary", summary), ("metrics", sim.metrics()),
                        ("messages", [asdict(m) for m in sim.messages.values()])):
        (args.output / f"{name}.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (args.output / "workload.json").write_text(workload.to_json())
    (args.output / "events.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in sim.log))
    (args.output / "topologies.jsonl").write_text("".join(
        json.dumps({"at": s.at, "topology": json.loads(s.topology.to_json())}, sort_keys=True) + "\n"
        for s in snapshots))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
