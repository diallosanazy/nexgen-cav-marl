"""Seeded diagnostic load sweep; run with python3 -m nexgen_sim.congestion_demo."""

import json

from . import Link, Simulator, Stream, Topology, poisson


def run_case(rate, *, seed=42):
    topology = Topology(["source-a", "source-b", "hub", "sink"],
                        {("source-a", "hub"): Link(100),
                         ("source-b", "hub"): Link(100),
                         ("hub", "sink"): Link(1000)})
    # Two independently configured streams share source-a.
    streams = [Stream(id, source, "sink", rate / 3, 5000, 0, 1_000_000)
               for id, source in [("a1", "source-a"), ("a2", "source-a"), ("b1", "source-b")]]
    workload = poisson(streams, seed=seed, topology=topology)
    sim = Simulator.from_topology(topology, lambda o: "sink" if o.node == "hub" else "hub",
                                  queue_capacity=4)
    workload.replay(sim)
    sim.run(1_000_000)
    result = sim.metrics()
    return {"seed": seed, "offered_messages_per_second": rate,
            "nominal_hub_load": rate / 1000,
            **{k: result[k] for k in ("generated", "delivered", "dropped", "expired", "pending",
                                     "throughput_per_second", "mean_delivered_latency_us", "drop_reasons")},
            "hub": result["nodes"]["hub"],
            "total_queue_residence_us": sum(m.queue_residence for m in sim.messages.values())}


def main():
    print(json.dumps([run_case(rate) for rate in (200, 600, 2000)], indent=2))


if __name__ == "__main__":
    main()
