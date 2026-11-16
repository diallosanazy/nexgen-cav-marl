# Unified Waymo–DES integration (Specification 003)

Waymo-derived mobility is part of the simulator's end-to-end research foundation.
Synthetic graphs remain regression and generalization fixtures. The integration
uses the existing DES clock, queues, traffic and lifecycle, not a second engine.

## Setup and validation

From the repository root, Python 3.11 or newer:

```sh
python3 -m venv /tmp/nexgen-integration-venv
/tmp/nexgen-integration-venv/bin/pip install -e './sp2[test]'
PYTHONDONTWRITEBYTECODE=1 /tmp/nexgen-integration-venv/bin/python -m pytest \
  -q -p no:cacheprovider --import-mode=importlib tests sp2/tests
```

Both suites contain a `test_topology.py`; importlib collection avoids a module-name
collision. The optional Waymo adapter requires the teammate's package and its
`google-crc32c` dependency. The core DES still uses only the standard library.

## Reproduce the real-data slice

The local shard is an uncompressed training shard, not the `training_20s` filename
expected by the teammate's bulk preparation script. The unified runner accepts its
actual path and uses timestamps/duration from the record.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/nexgen-integration-venv/bin/python -m nexgen_sim.waymo_run \
  --shard /home/marc/Documents/capstone2/waymo_data_local/training/uncompressed_scenario_training_training.tfrecord-00000-of-01000 \
  --scenario-id 31541266c6a045e --cohort-size 8 --selection-seed 42 \
  --radius 30 --traffic-seed 42 --rate 5 --lifetime 1000000 \
  --output /home/marc/Documents/capstone2/waymo_data_local/my-new-run
```

Choose a new output directory per run; existing runs are never overwritten.
Add `--frozen` for a diagnostic control that applies only the initial topology,
while preserving the same traffic and horizon. Scenario selection for this example
was diagnostic: it exhibits connectivity changes, not a representative test split.
The 30 m communication radius and fixed timing values are illustrative abstractions,
not calibrated radio parameters.

Other CLI options include `--transmission`, `--propagation`, `--processing`,
`--wait-delay` (all integer microseconds), and `--queue-capacity`.
Use `--streams streams.json` for a JSON list of explicit stream configurations:

```json
[{"id":"example","source":"2207","destination":"2189",
  "rate_per_second":5,"lifetime":1000000,"start":0,"stop":8999900}]
```

Endpoints must belong to the selected cohort. Without that file, the runner exports
explicit diagnostic pairs forming a ring of sorted selected vehicle IDs; each has
its own stream. This is not random destination selection. Stream generation windows
must end no later than the last trajectory timestamp. Lifetimes may extend beyond
the horizon; unfinished messages remain pending.

## Shared contract

- `TopologySnapshot(at, topology)` is independent of any dataset. `schedule_snapshots`
  validates the complete sequence before adding its events to the DES.
- `waymo_snapshots` reuses the teammate's positions/proximity functions. Integer
  Waymo IDs map bijectively to strings, with the original ID retained in metadata.
- A deterministic cohort is selected from tracks valid for every aligned frame.
  Node entry/exit is deliberately unsupported; movement and link changes are active.
  This cohort rule introduces selection bias that future experiments must report.
- Frame times become `round((timestamp - first_timestamp) * 1e6)`. Converted times
  must strictly increase. Positions/connectivity are held until the next frame;
  there is no interpolation or independent mobility tick loop.
- Equal-time priority is expiration, topology update, serialization finish, arrival,
  generation, wake, decision, service selection. Queues and reservations survive
  topology updates. Updates do not reset WAIT eligibility or preempt service.
- Each serialization captures its endpoints, finish time and propagation delay.
  A link disappearance strictly before finish permanently invalidates the attempt,
  even if the link returns. The slot stays reserved until finish, when a `link_break`
  drop is recorded unless expiration has already terminated the message.
- A link disappearing exactly at finish does not invalidate completed serialization.
  Later propagation survives connectivity changes. Changing delay metadata does not
  retime an attempt already underway. These are explicit model assumptions.
- All original FIFO/WAIT, permanent channel-failure drop, queue overflow, deadline
  and conservation rules remain in force. Link breaks are distinct from channel loss.

## Routing and future learning

`ShortestPathReference` reads the current graph through an explicitly privileged
provider. It recomputes a hop-shortest route at each decision, uses lexicographic
neighbor order to break ties and returns WAIT when unreachable. It has no future
trajectory access. It is not GPSR, AODV, or a decentralized learned actor.

The existing local `Observation` and per-message policy callback remain unchanged.
Positions and global metadata do not automatically leak into actor observations.
Decision times are retained in WAIT and transmission-start events. A future RL
wrapper must account for unequal elapsed decision intervals, define rewards and
critic inputs, and distinguish terminal outcomes from horizon truncation. No such
training wrapper or complete baseline protocol is claimed here.

## Artifacts and evidence

Each run exports:

- `config.json`: selected tracks, scenario, source frame alignment, timing/radius,
  explicit streams, seeds, Python version, shard size/path/SHA-256.
- `workload.json`: exact replayable generated-message schedule.
- `topologies.jsonl`: source snapshots with timestamps, positions and connectivity.
  Frozen control exports the source trajectory too; only its first snapshot is applied.
- `events.jsonl`: DES trace including topology changes, link invalidation, WAIT,
  forwarding and terminal events.
- `messages.json`, `metrics.json`, `summary.json`: lifecycle records and measurements.

`source_positions_changed` describes the input trajectory; `positions_changed`
reports whether changing positions were applied in this run. Connectivity change
counts are directed, so removal of one symmetric proximity link counts as two.
No radio interference, random channel-loss model, entry/exit semantics or RL training
has been introduced. See `../reports/WAYMO_INTEGRATION_2026-11-16.md` for measured results.
