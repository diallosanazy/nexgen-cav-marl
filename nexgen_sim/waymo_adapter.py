"""Waymo frames become DES snapshots; there is no separate mobility clock."""

from nexgen.topology import scenario_positions, build_proximity_graph

from .engine import Link
from .mobility import TopologySnapshot
from .topology import Topology


def waymo_snapshots(scenario, selected_ids, *, radius, transmission=1000, propagation=0):
    selected_ids = tuple(selected_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("select a nonempty unique vehicle cohort")
    link = Link(transmission, propagation)
    times = [round((t - scenario.timestamps[0]) * 1_000_000) for t in scenario.timestamps]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Waymo timestamps collapse or reverse at microsecond resolution")
    result = []
    for index, at in enumerate(times):
        positions = scenario_positions(scenario, index, selected_ids)
        graph = build_proximity_graph(positions, radius)
        topology = Topology([str(id) for id in selected_ids],
                            {(str(a), str(b)): link for a, neighbors in graph.items() for b in neighbors},
                            node_attributes={str(id): {"position": list(pos), "waymo_track_id": id}
                                             for id, pos in positions.items()})
        result.append(TopologySnapshot(at, topology))
    return tuple(result)
