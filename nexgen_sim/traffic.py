"""Generate traffic before simulation; replay never consults a policy or RNG."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random

from .engine import duration


@dataclass(frozen=True)
class Generation:
    id: str
    source: str
    destination: str
    at: int
    deadline: int

    def __post_init__(self):
        if any(not isinstance(x, str) for x in (self.id, self.source, self.destination)):
            raise ValueError("message and endpoint IDs must be strings")
        duration(self.at, "generation time")
        duration(self.deadline, "deadline")
        if self.deadline <= self.at:
            raise ValueError("deadline must follow generation")


@dataclass(frozen=True)
class Stream:
    id: str
    source: str
    destination: str
    rate_per_second: float
    lifetime: int
    start: int
    stop: int

    def __post_init__(self):
        if any(not isinstance(x, str) for x in (self.id, self.source, self.destination)):
            raise ValueError("stream and endpoint IDs must be strings")
        if (type(self.rate_per_second) not in (int, float) or
                not math.isfinite(self.rate_per_second) or self.rate_per_second < 0):
            raise ValueError("rate must be finite and nonnegative")
        duration(self.lifetime, "lifetime", positive=True)
        duration(self.start, "start")
        duration(self.stop, "stop")
        if self.stop < self.start:
            raise ValueError("stop must not precede start")


@dataclass(frozen=True)
class Workload:
    records: tuple[Generation, ...]

    def __post_init__(self):
        records = tuple(self.records)
        if any(not isinstance(r, Generation) for r in records):
            raise ValueError("workload requires Generation records")
        if len({r.id for r in records}) != len(records):
            raise ValueError("duplicate message IDs")
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda r: (r.at, r.id))))

    def validate(self, topology):
        nodes = set(topology.nodes)
        if any(r.source not in nodes or r.destination not in nodes for r in self.records):
            raise ValueError("workload references an unknown node")
        return self

    def replay(self, sim):
        self.validate(sim.topology)
        if any(r.at < sim.time or r.id in sim._reserved for r in self.records):
            raise ValueError("workload conflicts with simulator time or reserved IDs")
        for r in self.records:
            sim.generate(r.id, r.source, r.destination, at=r.at, deadline=r.deadline)

    def to_json(self):
        return json.dumps({"version": 1, "messages": [asdict(r) for r in self.records]},
                          sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text):
        try:
            data = json.loads(text)
            if data["version"] != 1:
                raise ValueError("unsupported workload version")
            return cls(tuple(Generation(**r) for r in data["messages"]))
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid workload JSON") from exc


def _message_id(stream_id, index):
    # Structured encoding avoids collisions even when IDs contain punctuation.
    return json.dumps([stream_id, index], ensure_ascii=False, separators=(",", ":"))


def periodic(stream_id, source, destination, *, interval, lifetime, start, stop):
    duration(interval, "interval", positive=True)
    Stream(stream_id, source, destination, 0, lifetime, start, stop)
    return Workload(tuple(Generation(_message_id(stream_id, i), source, destination,
                                     at, at + lifetime)
                          for i, at in enumerate(range(start, stop, interval))))


def poisson(streams, *, seed, topology):
    """Independent per-stream exponential arrivals; window excludes stop."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    streams = tuple(streams)
    if len({s.id for s in streams}) != len(streams):
        raise ValueError("duplicate stream IDs")
    nodes = set(topology.nodes)
    if any(s.source not in nodes or s.destination not in nodes for s in streams):
        raise ValueError("stream references unknown node")
    records = []
    for stream in sorted(streams, key=lambda s: s.id):
        if stream.rate_per_second == 0 or stream.start == stream.stop:
            continue
        digest = hashlib.sha256(json.dumps([seed, stream.id]).encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        elapsed = 0.0
        index = 0
        while True:
            gap = rng.expovariate(stream.rate_per_second) * 1_000_000
            next_elapsed = elapsed + gap
            if not math.isfinite(next_elapsed):
                break
            if next_elapsed <= elapsed:
                raise ValueError("arrival rate exceeds floating-point time resolution")
            elapsed = next_elapsed
            at = stream.start + math.ceil(elapsed)
            if at >= stream.stop:
                break
            records.append(Generation(_message_id(stream.id, index), stream.source,
                                      stream.destination, at, at + stream.lifetime))
            index += 1
    return Workload(tuple(records))
