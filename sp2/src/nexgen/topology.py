"""Dynamic proximity graphs derived from aligned vehicle positions."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .waymo import Scenario

Position = tuple[float, float]
Graph = dict[int, frozenset[int]]


@dataclass(frozen=True, slots=True)
class TopologyMetrics:
    node_count: int
    edge_count: int
    mean_degree: float
    maximum_degree: int
    density: float
    connected_components: int


def build_proximity_graph(
    positions: Mapping[int, Position], communication_radius: float
) -> Graph:
    """Build a symmetric, undirected graph using inclusive Euclidean radius."""
    if not math.isfinite(communication_radius) or communication_radius < 0:
        raise ValueError("communication_radius must be finite and non-negative")

    normalized_positions: dict[int, Position] = {}
    for node_id, position in positions.items():
        if len(position) != 2:
            raise ValueError(f"node {node_id} position must have exactly two values")
        x, y = position
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"node {node_id} position must be finite")
        normalized_positions[node_id] = (float(x), float(y))

    node_ids = sorted(normalized_positions)
    neighbors: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    radius_squared = communication_radius * communication_radius
    for index, left_id in enumerate(node_ids):
        left_x, left_y = normalized_positions[left_id]
        for right_id in node_ids[index + 1 :]:
            right_x, right_y = normalized_positions[right_id]
            dx = left_x - right_x
            dy = left_y - right_y
            if dx * dx + dy * dy <= radius_squared:
                neighbors[left_id].add(right_id)
                neighbors[right_id].add(left_id)
    return {node_id: frozenset(neighbors[node_id]) for node_id in node_ids}


def scenario_positions(
    scenario: Scenario,
    frame_index: int,
    selected_track_ids: Iterable[int] | None = None,
) -> dict[int, Position]:
    """Return the valid vehicle positions for one aligned motion frame."""
    if not 0 <= frame_index < scenario.aligned_frame_count:
        raise IndexError(
            f"frame_index {frame_index} is outside scenario "
            f"{scenario.scenario_id} ({scenario.aligned_frame_count} frames)"
        )

    selected = None if selected_track_ids is None else set(selected_track_ids)
    tracks_by_id = {track.track_id: track for track in scenario.vehicles}
    if selected is not None:
        missing_ids = selected.difference(tracks_by_id)
        if missing_ids:
            raise ValueError(
                f"selected vehicle IDs are absent from "
                f"{scenario.scenario_id}: {sorted(missing_ids)}"
            )

    positions: dict[int, Position] = {}
    for track in scenario.vehicles:
        if selected is not None and track.track_id not in selected:
            continue
        state = track.states[frame_index]
        if state.has_position:
            positions[track.track_id] = (state.x, state.y)
        elif selected is not None:
            raise ValueError(
                f"selected vehicle {track.track_id} is invalid at frame "
                f"{frame_index} in {scenario.scenario_id}"
            )
    return positions


def scenario_topology(
    scenario: Scenario,
    frame_index: int,
    communication_radius: float,
    selected_track_ids: Iterable[int] | None = None,
) -> Graph:
    """Build the proximity graph at one 100 ms Waymo motion frame."""
    return build_proximity_graph(
        scenario_positions(scenario, frame_index, selected_track_ids),
        communication_radius,
    )


def iter_scenario_topologies(
    scenario: Scenario,
    communication_radius: float,
    selected_track_ids: Iterable[int] | None = None,
) -> Iterable[Graph]:
    """Yield graphs in frame order; use scenario timestamps to schedule updates.

    The DES adapter holds each graph until the next timestamp. No independent
    fixed-tick mobility clock is implied by this iterator.
    """
    stable_selection = (
        None if selected_track_ids is None else tuple(selected_track_ids)
    )
    for frame_index in range(scenario.aligned_frame_count):
        yield scenario_topology(
            scenario,
            frame_index,
            communication_radius,
            stable_selection,
        )


def topology_metrics(graph: Mapping[int, Iterable[int]]) -> TopologyMetrics:
    """Measure undirected degree, density, and connected components."""
    node_ids = set(graph)
    adjacency: dict[int, set[int]] = {}
    for node_id, neighbors in graph.items():
        neighbor_set = set(neighbors)
        if node_id in neighbor_set:
            raise ValueError(f"graph has a self-link at node {node_id}")
        if not neighbor_set.issubset(node_ids):
            raise ValueError(f"graph has a neighbor missing from its nodes at {node_id}")
        adjacency[node_id] = neighbor_set
    for node_id, neighbors in adjacency.items():
        for neighbor_id in neighbors:
            if node_id not in adjacency[neighbor_id]:
                raise ValueError("graph must be symmetric and undirected")

    node_count = len(node_ids)
    degrees = [len(adjacency[node_id]) for node_id in node_ids]
    edge_count = sum(degrees) // 2
    possible_edges = node_count * (node_count - 1) // 2
    density = edge_count / possible_edges if possible_edges else 0.0
    mean_degree = sum(degrees) / node_count if node_count else 0.0

    unseen = set(node_ids)
    component_count = 0
    while unseen:
        component_count += 1
        start = unseen.pop()
        frontier = deque([start])
        while frontier:
            current = frontier.popleft()
            newly_seen = adjacency[current].intersection(unseen)
            unseen.difference_update(newly_seen)
            frontier.extend(newly_seen)

    return TopologyMetrics(
        node_count=node_count,
        edge_count=edge_count,
        mean_degree=mean_degree,
        maximum_degree=max(degrees, default=0),
        density=density,
        connected_components=component_count,
    )


def _undirected_edges(graph: Mapping[int, Iterable[int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for node_id, neighbors in graph.items():
        for neighbor_id in neighbors:
            if node_id != neighbor_id:
                edges.add((min(node_id, neighbor_id), max(node_id, neighbor_id)))
    return edges


def count_link_changes(
    previous_graph: Mapping[int, Iterable[int]],
    current_graph: Mapping[int, Iterable[int]],
) -> int:
    """Count undirected links added or removed between consecutive frames."""
    return len(_undirected_edges(previous_graph) ^ _undirected_edges(current_graph))
