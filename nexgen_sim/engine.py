"""Specification 001: static, single-copy, store-and-forward simulation."""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import heapq
from itertools import count
from typing import Callable


class Status(str, Enum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    DROPPED = "DROPPED"
    EXPIRED = "EXPIRED"


class _Wait(Enum):
    WAIT = "WAIT"


WAIT = _Wait.WAIT


def duration(value, name, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be an integer {'positive' if positive else 'nonnegative'} microsecond value")


@dataclass(frozen=True)
class Link:
    transmission: int
    propagation: int = 0

    def __post_init__(self):
        duration(self.transmission, "transmission", positive=True)
        duration(self.propagation, "propagation")


@dataclass
class Message:
    id: str
    source: str
    destination: str
    generated_at: int
    deadline: int
    location: str | None
    status: Status = Status.PENDING
    eligible_at: int = 0
    delivered_at: int | None = None
    attempts: int = 0
    history: list[str] = field(default_factory=list)
    queue_residence: int = 0
    deliberate_wait: int = 0


@dataclass(frozen=True)
class Observation:
    time: int
    node: str
    message_id: str
    destination: str
    age: int
    deadline: int
    neighbors: tuple[str, ...]
    queue_ids: tuple[str, ...]


@dataclass
class Node:
    queue: list[str] = field(default_factory=list)
    busy: str | None = None
    transmitting: bool = False
    queue_area: int = 0
    peak_queue: int = 0
    service_time: int = 0
    transmission_time: int = 0


class Simulator:
    """FIFO among eligible messages; directed links; no receiver contention.

    Event priority: expiration, transmission finish, arrival, generation,
    wake, service. Insertion sequence breaks remaining ties. Policy callbacks
    receive only an immutable local observation. Outcome callbacks are a
    deterministic channel test seam: True delivers, False permanently drops.
    """

    PRIORITY = {"expire": 0, "finish": 1, "arrive": 2,
                "generate": 3, "wake": 4, "decide": 5, "service": 6}

    def __init__(self, nodes, links: dict[tuple[str, str], Link],
                 policy: Callable[[Observation], object], *, wait_delay=5,
                 processing=0, queue_capacity=100, outcome=None):
        duration(wait_delay, "wait_delay", positive=True)
        duration(processing, "processing")
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")
        from .topology import Topology
        self.topology = Topology(nodes, links)
        self.nodes = {n: Node() for n in self.topology.nodes}
        self.links = self.topology.links
        self.policy = policy
        self.outcome = outcome if outcome is not None else lambda *_: True
        self.wait_delay, self.processing = wait_delay, processing
        self.queue_capacity = queue_capacity
        self.time = 0
        self.messages: dict[str, Message] = {}
        self.events = []
        self.log = []
        self._sequence = count()
        self._reserved = set()

    @classmethod
    def from_topology(cls, topology, policy, **kwargs):
        sim = cls(topology.nodes, topology.links, policy, **kwargs)
        sim.topology = topology
        return sim

    def _schedule(self, time, kind, *args):
        heapq.heappush(self.events, (time, self.PRIORITY[kind], next(self._sequence), kind, args))

    def _record(self, kind, message=None, **details):
        self.log.append(dict(time=self.time, event=kind, message=message, **details))

    def generate(self, id, source, destination, *, at=0, deadline=100):
        duration(at, "at")
        duration(deadline, "deadline")
        if at < self.time or deadline <= at:
            raise ValueError("generation must not be in the past; deadline must follow generation")
        if id in self._reserved or source not in self.nodes or destination not in self.nodes:
            raise ValueError("duplicate message ID or unknown endpoint")
        self._reserved.add(id)
        self._schedule(at, "generate", id, source, destination, deadline)

    def accounting(self):
        counts = Counter(m.status.value.lower() for m in self.messages.values())
        return {"generated": len(self.messages), **{s.value.lower(): counts[s.value.lower()] for s in Status}}

    def assert_invariants(self):
        c = self.accounting()
        assert c["generated"] == sum(c[k] for k in ("pending", "delivered", "dropped", "expired"))
        queued = [id for n in self.nodes.values() for id in n.queue]
        assert len(queued) == len(set(queued))
        for node, state in self.nodes.items():
            assert len(state.queue) <= self.queue_capacity
            for id in state.queue:
                m = self.messages[id]
                assert m.status == Status.PENDING and m.location == node
        for m in self.messages.values():
            if m.status == Status.PENDING and m.location is not None:
                assert m.id in self.nodes[m.location].queue
            if m.status != Status.PENDING:
                assert m.id not in queued
            if m.status == Status.DELIVERED:
                assert m.location == m.destination

    def run(self, until):
        duration(until, "until")
        if until < self.time:
            raise ValueError("cannot run backwards")
        while self.events and self.events[0][0] <= until:
            time, _, _, kind, args = heapq.heappop(self.events)
            self._advance_time(time)
            getattr(self, "_" + kind)(*args)
            for state in self.nodes.values():
                state.peak_queue = max(state.peak_queue, len(state.queue))
            self.assert_invariants()
        self._advance_time(until)
        return self.accounting()

    def _advance_time(self, time):
        elapsed = time - self.time
        for state in self.nodes.values():
            state.queue_area += len(state.queue) * elapsed
            state.service_time += elapsed if state.busy is not None else 0
            state.transmission_time += elapsed if state.transmitting else 0
            for id in state.queue:
                if id == state.busy:  # processing occupies buffer, but is not queueing
                    continue
                m = self.messages[id]
                m.queue_residence += elapsed
                m.deliberate_wait += max(0, min(time, m.eligible_at) - self.time)
        self.time = time

    def metrics(self):
        counts = self.accounting()
        delivered = [m for m in self.messages.values() if m.status == Status.DELIVERED]
        drops = Counter(e["reason"] for e in self.log if e["event"] == "dropped")
        return {**counts,
                "delivery_ratio": counts["delivered"] / counts["generated"] if counts["generated"] else 0.0,
                "throughput_per_second": counts["delivered"] * 1_000_000 / self.time if self.time else 0.0,
                "mean_delivered_latency_us": (sum(m.delivered_at - m.generated_at for m in delivered) / len(delivered)
                                               if delivered else None),
                "drop_reasons": dict(drops),
                "nodes": {n: {"mean_queue_occupancy": s.queue_area / self.time if self.time else 0.0,
                              "peak_queue_occupancy": s.peak_queue,
                              "service_utilization": s.service_time / self.time if self.time else 0.0,
                              "transmission_utilization": s.transmission_time / self.time if self.time else 0.0}
                          for n, s in self.nodes.items()},
                "messages": {m.id: {"queue_residence_us": m.queue_residence,
                                     "deliberate_wait_us": m.deliberate_wait,
                                     "eligible_queue_delay_us": m.queue_residence - m.deliberate_wait}
                             for m in self.messages.values()}}

    def _generate(self, id, source, destination, deadline):
        m = Message(id, source, destination, self.time, deadline, source,
                    eligible_at=self.time, history=[source])
        self.messages[id] = m
        self._record("generated", id, node=source)
        self._schedule(deadline, "expire", id)
        self._accept(m, source)

    def _accept(self, m, node):
        m.location = node
        if node == m.destination:
            m.status, m.delivered_at = Status.DELIVERED, self.time
            self._record("delivered", m.id, node=node)
        elif len(self.nodes[node].queue) >= self.queue_capacity:
            m.status = Status.DROPPED
            self._record("dropped", m.id, node=node, reason="queue_overflow")
        else:
            self.nodes[node].queue.append(m.id)
            m.eligible_at = self.time
            self._record("enqueued", m.id, node=node)
            self._schedule(self.time, "service", node)

    def _expire(self, id):
        m = self.messages[id]
        if m.status != Status.PENDING:
            return
        m.status = Status.EXPIRED
        if m.location is not None and id in self.nodes[m.location].queue:
            self.nodes[m.location].queue.remove(id)
            self._schedule(self.time, "service", m.location)
        self._record("expired", id)

    def _wake(self, id, node):
        if self.messages[id].status == Status.PENDING and id in self.nodes[node].queue:
            self._record("eligible", id, node=node)
            self._schedule(self.time, "service", node)

    def _service(self, node):
        state = self.nodes[node]
        if state.busy is not None:
            return
        m = next((self.messages[id] for id in state.queue
                  if self.messages[id].eligible_at <= self.time), None)
        if m is None:
            return
        state.busy = m.id
        self._record("processing_started", m.id, node=node)
        self._schedule(self.time + self.processing, "decide", m.id, node)

    def _decide(self, id, node):
        state = self.nodes[node]
        m = self.messages[id]
        if m.status != Status.PENDING:
            state.busy = None
            self._schedule(self.time, "service", node)
            return
        obs = Observation(self.time, node, m.id, m.destination,
                          self.time - m.generated_at, m.deadline,
                          self.topology.neighbors(node), tuple(state.queue))
        action = self.policy(obs)
        if action == WAIT:
            state.busy = None
            m.eligible_at = self.time + self.wait_delay
            self._record("wait", m.id, node=node, eligible_at=m.eligible_at)
            self._schedule(m.eligible_at, "wake", m.id, node)
            self._schedule(self.time, "service", node)
            return
        if action not in obs.neighbors:
            raise ValueError(f"invalid next hop {action!r} from {node!r}")
        state.busy = m.id
        state.queue.remove(m.id)
        state.transmitting = True
        m.location = None
        m.attempts += 1
        self._record("service_started", m.id, source=node, target=action,
                     transmission_at=self.time)
        self._schedule(self.time + self.links[node, action].transmission,
                       "finish", m.id, node, action)

    def _finish(self, id, source, target):
        self.nodes[source].busy = None
        self.nodes[source].transmitting = False
        self._record("service_finished", id, source=source, target=target)
        self._schedule(self.time, "service", source)
        m = self.messages[id]
        if m.status != Status.PENDING:
            return
        self._schedule(self.time + self.links[source, target].propagation,
                       "arrive", id, source, target)

    def _arrive(self, id, source, target):
        m = self.messages[id]
        if m.status != Status.PENDING:
            return
        if not self.outcome(id, source, target, m.attempts):
            m.status = Status.DROPPED
            self._record("dropped", id, source=source, target=target, reason="transmission_failure")
            return
        m.history.append(target)
        self._record("arrived", id, node=target)
        self._accept(m, target)
