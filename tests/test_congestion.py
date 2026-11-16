import unittest
from nexgen_sim import Generation, Link, Simulator, Topology, WAIT, Workload, periodic


class CongestionTests(unittest.TestCase):
    def graph(self):
        return Topology(["A", "D"], {("A", "D"): Link(10)})

    def test_hand_calculated_queue_and_utilization(self):
        sim = Simulator.from_topology(self.graph(), lambda _: "D", queue_capacity=2)
        Workload((Generation("one", "A", "D", 0, 100),
                  Generation("two", "A", "D", 1, 100),
                  Generation("three", "A", "D", 2, 100),
                  Generation("overflow", "A", "D", 3, 100))).replay(sim)
        sim.run(30)
        m = sim.metrics()
        self.assertEqual(m["delivered"], 3)
        self.assertEqual(m["drop_reasons"], {"queue_overflow": 1})
        self.assertEqual(m["nodes"]["A"]["mean_queue_occupancy"], 27 / 30)
        self.assertEqual(m["nodes"]["A"]["peak_queue_occupancy"], 2)
        self.assertEqual(m["nodes"]["A"]["transmission_utilization"], 1)
        self.assertEqual(m["messages"]["two"]["queue_residence_us"], 9)
        self.assertEqual(m["messages"]["three"]["queue_residence_us"], 18)
        self.assertEqual(m["throughput_per_second"], 100_000)

    def test_processing_propagation_horizon_and_resume(self):
        metrics = []
        for split in (False, True):
            sim = Simulator(["A", "D"], {("A", "D"): Link(10, 20)}, lambda _: "D", processing=2)
            sim.generate("M", "A", "D")
            if split:
                sim.run(5)
            sim.run(20)
            metrics.append(sim.metrics())
        self.assertEqual(*metrics)
        self.assertEqual(metrics[0]["pending"], 1)
        self.assertEqual(metrics[0]["nodes"]["A"]["service_utilization"], 12 / 20)
        self.assertEqual(metrics[0]["nodes"]["A"]["transmission_utilization"], 10 / 20)

    def test_wait_delay_separated_from_eligible_queue_delay(self):
        sim = Simulator.from_topology(self.graph(),
                                      lambda o: WAIT if o.message_id == "one" and o.time == 0 else "D")
        sim.generate("one", "A", "D")
        sim.generate("two", "A", "D")
        sim.run(20)
        self.assertEqual(sim.metrics()["messages"]["one"],
                         dict(queue_residence_us=10, deliberate_wait_us=5, eligible_queue_delay_us=5))

    def test_expired_and_pending_waits_are_measured(self):
        sim = Simulator.from_topology(self.graph(), lambda _: WAIT, wait_delay=20)
        sim.generate("expired", "A", "D", deadline=5)
        sim.generate("pending", "A", "D", deadline=50)
        sim.run(10)
        m = sim.metrics()
        self.assertEqual((m["expired"], m["pending"]), (1, 1))
        self.assertEqual(m["messages"]["expired"]["queue_residence_us"], 5)
        self.assertEqual(m["messages"]["pending"]["queue_residence_us"], 10)

    def test_deterministic_offered_load_changes_outcomes(self):
        results = []
        for interval in (40, 20, 5):
            sim = Simulator.from_topology(self.graph(), lambda _: "D", queue_capacity=2)
            periodic("load", "A", "D", interval=interval, lifetime=1000, start=0, stop=200).replay(sim)
            sim.run(200)
            results.append(sim.metrics())
        self.assertEqual([r["nodes"]["A"]["transmission_utilization"] for r in results], [0.25, 0.5, 1])
        self.assertEqual([r["delivered"] for r in results[:2]], [5, 10])
        self.assertGreater(results[2]["dropped"], 0)
        self.assertGreater(results[2]["pending"], 0)
        self.assertEqual(results[2]["throughput_per_second"], 100_000)

    def test_congestion_expires_messages(self):
        sim = Simulator.from_topology(self.graph(), lambda _: "D")
        periodic("load", "A", "D", interval=2, lifetime=15, start=0, stop=50).replay(sim)
        sim.run(100)
        m = sim.metrics()
        self.assertGreater(m["expired"], 0)
        self.assertGreater(m["delivered"], 0)
        self.assertEqual(m["generated"], m["expired"] + m["delivered"])

    def test_zero_horizon(self):
        sim = Simulator.from_topology(self.graph(), lambda _: "D")
        self.assertEqual(sim.metrics()["throughput_per_second"], 0)
        self.assertIsNone(sim.metrics()["mean_delivered_latency_us"])
