# Senior Project II

This folder contains the current Waymo mobility and scenario-preparation tools for
the V2V routing study.

## Current implementation

- `src/nexgen/waymo.py` reads Waymo Open Motion records and aligns timestamps with
  vehicle tracks.
- `src/nexgen/splits.py` checks full-episode eligibility, selects reproducible
  vehicle samples, and assigns scenario-level splits.
- `src/nexgen/topology.py` builds proximity graphs and computes topology metrics.
- `scripts/prepare_waymo_scenarios.py` creates a scenario eligibility and split
  CSV from local Waymo shards.

The routing simulator and its integration with Waymo mobility are not included in
this branch.

## Setup and tests

Requires Python 3.11 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest
```

## Prepare scenarios

Place the uncompressed Waymo `training_20s` shards in `data/waymo/raw`, then run:

```sh
python scripts/prepare_waymo_scenarios.py \
  --data-dir data/waymo/raw \
  --output-dir data/waymo \
  --first-shard 0 \
  --last-shard 9
```

Waymo data and generated selection files remain local and are excluded from Git.
