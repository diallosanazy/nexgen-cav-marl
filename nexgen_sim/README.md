# Specification 001: deterministic simulator foundation

Run from the repository root with Python 3.10 or newer (validated on 3.14.4):

```sh
python3 -m unittest discover -s tests -v
python3 -m nexgen_sim
```

The demo prints complete traces and accounting for two scenarios. No dependencies,
training, file writes, or imports of historical experiments are required.

## Model and interfaces

`Simulator(nodes, links, policy, ...)` owns the clock, message registry, node
queues, event heap and log. Node IDs are arbitrary strings. Links are directed;
bidirectional connectivity requires two links. `Link(transmission, propagation)`
uses fixed integer microseconds for equal-sized messages. Size/rate-derived service
times and stochastic loss are deferred. `generate(id, source, destination, at=...,
deadline=...)` schedules deterministic traffic; deadlines are absolute times.

Policies receive a frozen `Observation` with current time, local node, selected
message ID/destination/age/deadline, valid neighbors and local queue IDs. They return
a neighboring node ID or `WAIT`. The diagnostic scripted policy may encode a route;
this is not a conventional protocol or evidence of decentralized learning.
Invalid actions raise `ValueError` and abort the run; they are programming errors,
not simulated drops. Do not continue an aborted simulator.

FIFO selects the earliest enqueued eligible message. A single node service slot
covers processing and serialization, including processing for WAIT decisions.
Observations are constructed after processing, at decision time. A processing
message remains in the queue and counts toward capacity; serialization removes it
from the queue. Propagation does not occupy the sender slot. Other nodes operate
independently. No wireless interference, receiver contention, or retransmissions
are modeled.

WAIT keeps the message pending and in its original queue position. It sets a
positive eligibility delay from decision time, releases the service slot, and
allows other eligible messages to proceed. Eligibility does not preempt active
service. Deadline expiration wins ties with all other events. Expiration during
processing or transmission does not reclaim the occupied resource early; the
scheduled completion releases it, without forwarding the expired message.

Equal-time event order is expiration, transmission finish, arrival, generation,
WAIT wake, routing decision, service selection; insertion sequence breaks remaining
ties. Nodes and neighbors are sorted so their container order cannot change policy
inputs. Equal-time generation order follows explicit `generate` call order.

`run(until)` processes events at or before the inclusive fixed horizon, preserves
pending messages and future events, and can resume. Only actual generation events
enter accounting. Each generated message has one terminal or pending status.
Source-equals-destination is immediate delivery with zero attempts. Destination
delivery bypasses forwarding queue admission. Full queues drop arriving messages.
The optional pure `outcome(id, source, target, attempt)` callback supplies scripted
success/failure at arrival time; failure is a permanent drop, with no retry.

`messages`, `accounting()` and `log` support inspection. Route history records
successful arrivals; `attempts` counts transmission starts, not WAIT or processing.
Latency is delivery time minus generation time. Event records retain drop reasons,
WAIT eligibility times, service endpoints and arrival/delivery times. Conservation
and queue/status consistency are asserted after every processed event in normal
Python execution (do not use `python -O` for validation).

## Acceptance results

- Line: 3 µs processing + 10 µs transmission + 2 µs propagation per hop;
  A → B → C delivers at 30 µs, with two attempts and one generated/delivered message.
- WAIT: M1 waits at time 0 until eligible at 5; M2 transmits at 0, M1 at 10,
  and M3 at 20. Deliveries occur at 10, 20, and 30 µs respectively.
- Twelve deterministic tests cover these scenarios, reproducibility, failure,
  overflow, expiry during service and WAIT, expiry/arrival ties, fixed horizons,
  independent senders, arbitrary string IDs, invalid actions and duplicate IDs.

## Limits and next milestone

This is a small reference implementation, not a performance benchmark or RL
environment. Timing constants in examples are test fixtures, not calibrated radio
parameters. Specification 001 did not include stochastic traffic or metric aggregation;
Specification 002 adds these below. Random channels, mobility, learned policies,
GPSR, AODV, critic interfaces and engine checkpoint serialization remain deferred. Global engine state includes the event heap and service reservations;
reproduction also requires any state held by supplied policy/outcome callbacks.

Development continues in the separate package without waiting for teammate changes.
Reconcile those changes when available; preserve historical experiments in the meantime.
Independently seeded channel outcomes remain a later milestone.

# Specification 002: topology and replayable traffic

Run the seeded three-stream bottleneck demonstration:

```sh
python3 -m nexgen_sim.congestion_demo
```

The one-second scenarios offer 200, 600 and 2,000 messages/second to a hub with
1,000 transmissions/second capacity, four queue places and 5,000 µs message
lifetimes. Two streams originate at the same node, and a third at another node.
The example prints all terminal/pending counts, throughput, delivered latency,
queue occupancy and utilization. These are diagnostic cases, not calibrated
wireless experiments or conclusions about a routing algorithm.

## Construct, save and load a graph and workload

```python
from pathlib import Path
from nexgen_sim import Link, Simulator, Stream, Topology, Workload, poisson

graph = Topology(
    ["vehicle-10", "hub", "sink", "isolated"],
    {("vehicle-10", "hub"): Link(100, 2), ("hub", "sink"): Link(1000, 2)},
    node_attributes={"vehicle-10": {"position": [2, 4]}},
    edge_attributes={("hub", "sink"): {"bandwidth_bps": 1000000}},
)
Path("topology.json").write_text(graph.to_json())
graph = Topology.from_json(Path("topology.json").read_text())

streams = [Stream("updates", "vehicle-10", "sink", rate_per_second=300,
                  lifetime=5000, start=0, stop=1000000)]
workload = poisson(streams, seed=42, topology=graph)
Path("workload.json").write_text(workload.to_json())
workload = Workload.from_json(Path("workload.json").read_text())

sim = Simulator.from_topology(graph, lambda obs: obs.neighbors[0])
workload.replay(sim)
sim.run(1000000)
print(sim.metrics())
```

Keep diagnostic policies appropriate to the graph: this example's policy assumes
that every forwarding node has a next hop. An isolated node requires WAIT or another
explicit policy decision; isolation does not cause the loader to reject the graph.

Topology holds static data, not queues. Sorted outgoing-neighbor tuples avoid an
all-edge scan on each decision; read-only edge lookup retains Link values. Arbitrary
string IDs are preserved. Parallel duplicate edges and self edges are rejected;
undirected graphs require both directed edges. Attribute access returns copies.
The JSON version-1 format uses explicit node/edge records, timing fields named
`transmission_us` and `propagation_us`, and separate `metadata` objects. **Only those
timing fields affect communication.** Position, bandwidth and loss metadata neither
change the network model nor appear automatically in policy observations. The
legacy `Simulator(nodes, links, policy)` constructor remains supported.

`Generation(id, source, destination, at, deadline)` records arbitrary schedules.
`Workload(records)` validates unique IDs and sorts by `(at, id)`; this order controls
simultaneous traffic admission. `periodic(stream_id, source, destination, interval=...,
lifetime=..., start=..., stop=...)` creates integer-time deterministic traffic.
Combine streams with `Workload(tuple_of_all_records)`. Replay validates all endpoints,
times and ID conflicts before scheduling anything. Unreachable endpoints are legal;
unknown endpoints are not. Scheduled events after the run horizon are not generated
messages until their time is reached.

Poisson streams each have a unique stable ID, explicit pair, independent rate,
lifetime and half-open `[start, stop)` generation window. The first arrival follows
an exponential gap, not an automatic arrival at start. A master integer seed plus
stream ID derives a SHA-256 seed for a private standard-library RNG. Stream order,
other streams and policy decisions cannot consume its random draws. Zero rate is
valid and produces no messages. Continuous gaps accumulate before ceiling to integer
microseconds; multiple arrivals at the same timestamp are legal. Arrivals quantized
to stop are excluded. This discretization matters at extremely high rates: it is
not a continuous-time simulator. No positive minimum gap is artificially imposed.
Deadline is the quantized generation time plus the stream's lifetime.

Preserve exported workload JSON for cross-runtime replay: seed-based regeneration
is tested on Python 3.14.4, not promised across all Python/RNG versions. Workloads
are materialized in memory; large experiments will need separate scale validation.
Traffic and topology generators remain independent. Random graph generation,
random endpoints, bursty traffic and time-varying sources are future extensions.

## Measurement definitions

`metrics()` reports observations over `[0, current_time]` without draining messages:

- Generated, delivered, dropped, expired and pending; delivery ratio uses generated.
- Throughput is delivered messages / elapsed seconds. No deliveries yields a null
  mean delivered latency, not a fictitious zero latency.
- Mean queue occupancy is the time integral of buffer occupancy / elapsed time;
  peak occupancy includes instantaneous equal-time admissions. Processing messages
  still occupy a buffer slot, matching the established capacity rule.
- Service utilization counts processing plus serialization, including non-preemptive
  service after expiry. Transmission utilization counts serialization only, not
  propagation. WAIT eligibility delays do not occupy the service slot.
- Per-message queue residence includes deliberate WAIT and eligible waiting,
  excludes processing, and includes observed pending residence up to the horizon.
  `deliberate_wait_us` measures ineligible residence; `eligible_queue_delay_us`
  measures the remainder. Neither should be confused with delivered-only latency.
- Drop reasons distinguish queue overflow from transmission failure.

Measurements integrate actual state between events and through the final horizon.
Splitting a run into multiple `run()` calls preserves metrics. Zero elapsed time
produces zero utilization/throughput. The nominal offered hub load used by the demo
is aggregate routed arrival rate times `(processing + transmission) / 1e6`.
This is not a universal stability formula with WAIT, varying routes or losses.

Current limitations: full histories, workloads and invariant checks prioritize
correctness over scale. No random channel model, dynamic graph changes, active
bandwidth model, protocol baseline or learning algorithm is included. Specification
002 changes neither the established service rules nor the research questions.
