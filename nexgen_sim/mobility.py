"""Dataset-independent snapshots and a current-graph reference policy."""

from collections import deque
from dataclasses import dataclass

from .engine import WAIT, duration
from .topology import Topology


@dataclass(frozen=True)
class TopologySnapshot:
    at: int
    topology: Topology

    def __post_init__(self):
        duration(self.at, "snapshot time")


def schedule_snapshots(sim, snapshots):
    snapshots = tuple(snapshots)
    times = [s.at for s in snapshots]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("snapshot timestamps must strictly increase")
    if any(s.at < sim.time or s.at in sim._topology_times or
           tuple(s.topology.nodes) != tuple(sim.nodes) for s in snapshots):
        raise ValueError("invalid snapshot time or changing node cohort")
    for snapshot in snapshots:
        sim.schedule_topology(snapshot.topology, at=snapshot.at)


class ShortestPathReference:
    """Explicit full-current-graph privilege, no access to future snapshots.

    Not a decentralized actor or GPSR/AODV implementation. Equal-length paths
    use lexicographic node order. No reachable path means strategic WAIT.
    """

    def __init__(self, topology_provider):
        self.topology_provider = topology_provider

    def __call__(self, observation):
        topology = self.topology_provider()
        seen = {observation.node}
        queue = deque((neighbor, neighbor) for neighbor in topology.neighbors(observation.node))
        while queue:
            node, first = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            if node == observation.destination:
                return first
            queue.extend((neighbor, first) for neighbor in topology.neighbors(node) if neighbor not in seen)
        return WAIT
