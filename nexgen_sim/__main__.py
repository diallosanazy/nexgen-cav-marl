"""Run both deterministic acceptance scenarios and print their full traces."""

import json
from . import Link, Simulator, WAIT


def report(name, sim):
    print(json.dumps({"scenario": name, "accounting": sim.accounting(),
                      "messages": [{"id": m.id, "status": m.status.value,
                                    "delivered_at_us": m.delivered_at,
                                    "transmission_count": m.attempts,
                                    "route": m.history} for m in sim.messages.values()],
                      "events": sim.log}, indent=2))


def main():
    line = Simulator(["A", "B", "C"],
                     {("A", "B"): Link(10, 2), ("B", "C"): Link(10, 2)},
                     lambda obs: {"A": "B", "B": "C"}[obs.node], processing=3)
    line.generate("M1", "A", "C")
    line.run(30)
    report("line", line)

    waiting = Simulator(["A", "B"], {("A", "B"): Link(10)},
                        lambda obs: WAIT if obs.message_id == "M1" and obs.time == 0 else "B",
                        wait_delay=5)
    for id in ("M1", "M2", "M3"):
        waiting.generate(id, "A", "B")
    waiting.run(30)
    report("eligible_fifo", waiting)


if __name__ == "__main__":
    main()
