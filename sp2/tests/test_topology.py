import pytest

from nexgen.splits import (
    assign_scenario_splits,
    fully_valid_vehicle_ids,
    select_vehicle_ids,
)
from nexgen.topology import (
    build_proximity_graph,
    count_link_changes,
    iter_scenario_topologies,
    scenario_topology,
    topology_metrics,
)
from nexgen.waymo import MotionState, Scenario, VehicleTrack


def _state(x: float, y: float, valid: bool = True) -> MotionState:
    return MotionState(
        valid=valid,
        x=x if valid else None,
        y=y if valid else None,
        velocity_x=0.0,
        velocity_y=0.0,
    )


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="topology-scene",
        timestamps=(0.0, 0.1, 0.2),
        vehicles=(
            VehicleTrack(1, (_state(0.0, 0.0), _state(0.0, 0.0), _state(0.0, 0.0))),
            VehicleTrack(2, (_state(1.0, 0.0), _state(2.0, 0.0), _state(3.0, 0.0))),
            VehicleTrack(3, (_state(4.0, 0.0), _state(4.0, 0.0, False), _state(4.0, 0.0))),
        ),
        total_track_count=3,
        raw_timestamp_count=3,
        track_state_count_min=3,
        track_state_count_max=3,
        current_time_index=None,
    )


def test_proximity_graph_is_symmetric_and_includes_radius_boundary() -> None:
    graph = build_proximity_graph(
        {4: (0.0, 0.0), 2: (1.0, 0.0), 9: (2.1, 0.0)},
        communication_radius=1.0,
    )

    assert graph == {
        2: frozenset({4}),
        4: frozenset({2}),
        9: frozenset(),
    }


@pytest.mark.parametrize("radius", [-1.0, float("nan"), float("inf")])
def test_proximity_graph_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="communication_radius"):
        build_proximity_graph({1: (0.0, 0.0)}, radius)


def test_topology_metrics_count_degree_density_and_components() -> None:
    graph = build_proximity_graph(
        {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (4.0, 0.0)},
        communication_radius=1.0,
    )

    metrics = topology_metrics(graph)

    assert metrics.node_count == 3
    assert metrics.edge_count == 1
    assert metrics.mean_degree == pytest.approx(2 / 3)
    assert metrics.maximum_degree == 1
    assert metrics.density == pytest.approx(1 / 3)
    assert metrics.connected_components == 2


def test_scenario_graph_excludes_invalid_nodes_and_counts_link_churn() -> None:
    scenario = _scenario()
    selected = fully_valid_vehicle_ids(scenario)

    first = scenario_topology(scenario, 0, 2.0)
    middle = scenario_topology(scenario, 1, 2.0)

    assert first == {1: frozenset({2}), 2: frozenset({1}), 3: frozenset()}
    assert middle == {1: frozenset({2}), 2: frozenset({1})}
    assert count_link_changes(first, middle) == 0
    assert tuple(iter_scenario_topologies(scenario, 2.0, selected))[1] == {
        1: frozenset({2}),
        2: frozenset({1}),
    }


def test_fixed_scale_selection_is_seeded_and_nested() -> None:
    scenario = _scenario()

    eight = select_vehicle_ids(scenario, 1, seed=4)
    twenty_error = None
    try:
        select_vehicle_ids(scenario, 2, seed=4)
    except ValueError as error:
        twenty_error = error

    assert eight == (1,)
    assert twenty_error is None
    twenty = select_vehicle_ids(scenario, 2, seed=4)
    assert set(eight).issubset(twenty)
    with pytest.raises(ValueError, match="found 2"):
        select_vehicle_ids(scenario, 3, seed=4)


def test_scenario_splits_are_deterministic_and_size_stratified() -> None:
    timestamps = (0.0, 0.1)
    eligible_scenarios = [
        Scenario(
            scenario_id=f"large-{index:02d}",
            timestamps=timestamps,
            vehicles=tuple(
                VehicleTrack(
                    track_id=track_id,
                    states=(_state(float(track_id), 0.0),) * len(timestamps),
                )
                for track_id in range(50)
            ),
            total_track_count=50,
            raw_timestamp_count=2,
            track_state_count_min=2,
            track_state_count_max=2,
            current_time_index=None,
        )
        for index in range(20)
    ]

    first = assign_scenario_splits(eligible_scenarios, seed=15)
    second = assign_scenario_splits(eligible_scenarios, seed=15)

    assert first == second
    assert set(first) == {scenario.scenario_id for scenario in eligible_scenarios}
    assert {
        split: sum(value == split for value in first.values())
        for split in ("train", "development", "held_out")
    } == {"train": 14, "development": 3, "held_out": 3}
