import unittest

from nexgen_sim import Link, Simulator, Status, WAIT


class SimulatorTests(unittest.TestCase):
    def line(self, **kwargs):
        return Simulator(["A", "B", "C"],
                         {("A", "B"): Link(10, 2), ("B", "C"): Link(10, 2)},
                         lambda obs: {"A": "B", "B": "C"}[obs.node], **kwargs)

    def test_line_delivery(self):
        sim = self.line(processing=3)
        sim.generate("M1", "A", "C")
        sim.run(30)
        m = sim.messages["M1"]
        self.assertEqual((m.delivered_at, m.attempts, m.history), (30, 2, ["A", "B", "C"]))
        self.assertEqual(sim.accounting(), dict(generated=1, pending=0, delivered=1, dropped=0, expired=0))
        self.assertEqual([(e["time"], e["event"]) for e in sim.log], [
            (0, "generated"), (0, "enqueued"), (0, "processing_started"), (3, "service_started"),
            (13, "service_finished"), (15, "arrived"), (15, "enqueued"),
            (15, "processing_started"), (18, "service_started"), (28, "service_finished"),
            (30, "arrived"), (30, "delivered")])

    def test_wait_eligible_fifo_and_fresh_observation(self):
        observations = []
        def policy(obs):
            observations.append(obs)
            return WAIT if obs.message_id == "M1" and obs.time == 0 else "B"
        sim = Simulator(["A", "B"], {("A", "B"): Link(10)}, policy, wait_delay=5)
        for id in ("M1", "M2", "M3"):
            sim.generate(id, "A", "B")
        sim.run(30)
        starts = [(e["time"], e["message"]) for e in sim.log if e["event"] == "service_started"]
        self.assertEqual(starts, [(0, "M2"), (10, "M1"), (20, "M3")])
        m = sim.messages["M1"]
        self.assertEqual((m.delivered_at, m.attempts, m.history, m.deadline), (20, 1, ["A", "B"], 100))
        self.assertEqual([(o.time, o.age) for o in observations if o.message_id == "M1"], [(0, 0), (10, 10)])

    def test_expiration_precedes_arrival(self):
        sim = self.line()
        sim.generate("M", "A", "C", deadline=24)
        sim.run(100)
        self.assertEqual(sim.messages["M"].status, Status.EXPIRED)
        self.assertIsNone(sim.messages["M"].delivered_at)

    def test_wait_expiration_and_positive_time(self):
        sim = Simulator(["A", "B"], {}, lambda _: WAIT, wait_delay=5)
        sim.generate("M", "A", "B", deadline=12)
        sim.run(50)
        self.assertEqual([e["time"] for e in sim.log if e["event"] == "wait"], [0, 5, 10])
        self.assertEqual(sim.messages["M"].status, Status.EXPIRED)
        self.assertEqual(sim.messages["M"].attempts, 0)
        with self.assertRaises(ValueError):
            Simulator([], {}, lambda _: WAIT, wait_delay=0)

    def test_permanent_failure(self):
        sim = self.line(outcome=lambda *_: False)
        sim.generate("M", "A", "C")
        sim.run(100)
        self.assertEqual(sim.messages["M"].status, Status.DROPPED)
        self.assertEqual(sim.messages["M"].attempts, 1)
        self.assertEqual(sim.messages["M"].history, ["A"])

    def test_overflow_counts_waiting_message(self):
        sim = Simulator(["A", "B"], {}, lambda _: WAIT, queue_capacity=1)
        sim.generate("M1", "A", "B")
        sim.generate("M2", "A", "B", at=1)
        sim.run(2)
        self.assertEqual(sim.accounting(), dict(generated=2, pending=1, delivered=0, dropped=1, expired=0))
        self.assertEqual(sim.log[-1]["reason"], "queue_overflow")

    def test_expired_transmission_holds_slot_until_finish(self):
        sim = Simulator(["A", "B"], {("A", "B"): Link(10, 20)}, lambda _: "B")
        sim.generate("M1", "A", "B", deadline=5)
        sim.generate("M2", "A", "B")
        sim.run(40)
        self.assertEqual(sim.messages["M1"].status, Status.EXPIRED)
        self.assertEqual(sim.messages["M2"].delivered_at, 40)
        starts = [e["time"] for e in sim.log if e["event"] == "service_started"]
        self.assertEqual(starts, [0, 10])

    def test_processing_applies_to_wait_and_expiry(self):
        sim = Simulator(["A", "B"], {}, lambda _: WAIT, processing=3, wait_delay=5)
        sim.generate("M", "A", "B", deadline=10)
        sim.run(30)
        self.assertEqual([e["time"] for e in sim.log if e["event"] == "wait"], [3])
        self.assertEqual(sim.nodes["A"].busy, None)

    def test_repeatability_and_horizon(self):
        logs = []
        for _ in range(2):
            sim = self.line()
            sim.generate("M", "A", "C")
            sim.generate("future", "A", "C", at=50)
            sim.run(11)
            self.assertEqual(sim.accounting()["pending"], 1)
            self.assertEqual(sim.accounting()["generated"], 1)
            sim.run(24)
            logs.append(sim.log)
        self.assertEqual(*logs)

    def test_invalid_hop_fails_loudly(self):
        sim = Simulator(["A", "B"], {}, lambda _: "B")
        sim.generate("M", "A", "B")
        with self.assertRaises(ValueError):
            sim.run(10)

    def test_independent_nodes_overlap_and_string_ids(self):
        sim = Simulator(["car-9", "hub", "destination"],
                        {("car-9", "destination"): Link(10), ("hub", "destination"): Link(10)},
                        lambda _: "destination")
        sim.generate("one", "car-9", "destination")
        sim.generate("two", "hub", "destination")
        sim.run(10)
        self.assertEqual(sim.accounting()["delivered"], 2)

    def test_source_equals_destination_and_duplicate_id(self):
        sim = self.line()
        sim.generate("M", "A", "A")
        with self.assertRaises(ValueError):
            sim.generate("M", "A", "B")
        sim.run(0)
        self.assertEqual(sim.messages["M"].delivered_at, 0)
        self.assertEqual(sim.messages["M"].attempts, 0)


if __name__ == "__main__":
    unittest.main()
