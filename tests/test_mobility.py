import unittest

from nexgen_sim import Link, Simulator, Topology, WAIT
from nexgen_sim.mobility import ShortestPathReference, TopologySnapshot, schedule_snapshots


class MobilityTests(unittest.TestCase):
    def graph(self, connected=True, propagation=0):
        return Topology(["A", "B"], {("A", "B"): Link(10, propagation)} if connected else {})

    def sim(self, propagation=0):
        sim = Simulator.from_topology(self.graph(propagation=propagation), lambda _: WAIT)
        sim.policy = ShortestPathReference(lambda: sim.topology)
        return sim

    def test_break_and_reconnect_does_not_rescue_attempt(self):
        sim = self.sim()
        sim.generate("M", "A", "B")
        sim.schedule_topology(self.graph(False), at=4)
        sim.schedule_topology(self.graph(), at=6)
        sim.run(9)
        self.assertTrue(sim.nodes["A"].transmitting)
        self.assertEqual(sim.accounting()["pending"], 1)
        sim.run(10)
        self.assertEqual(sim.metrics()["drop_reasons"], {"link_break": 1})
        self.assertEqual(sim.nodes["A"].transmission_time, 10)

    def test_break_at_finish_and_during_propagation_survives(self):
        for at in (10, 12):
            sim = self.sim(propagation=5)
            sim.generate("M", "A", "B")
            sim.schedule_topology(self.graph(False), at=at)
            sim.run(15)
            self.assertEqual(sim.messages["M"].delivered_at, 15)

    def test_expiry_wins_break_and_finish(self):
        sim = self.sim()
        sim.generate("M", "A", "B", deadline=10)
        sim.schedule_topology(self.graph(False), at=5)
        sim.run(20)
        self.assertEqual(sim.accounting()["expired"], 1)
        self.assertEqual(sim.accounting()["dropped"], 0)

    def test_moving_vs_frozen_and_wait_eligibility(self):
        counts = []
        for moving in (False, True):
            sim = Simulator.from_topology(self.graph(False), lambda _: WAIT, wait_delay=5)
            sim.policy = ShortestPathReference(lambda: sim.topology)
            sim.generate("M", "A", "B", deadline=30)
            if moving:
                sim.schedule_topology(self.graph(), at=3)
            sim.run(30)
            counts.append(sim.accounting())
            if moving:
                starts = [e["time"] for e in sim.log if e["event"] == "service_started"]
                self.assertEqual(starts, [5])
        self.assertEqual(counts[0]["expired"], 1)
        self.assertEqual(counts[1]["delivered"], 1)

    def test_update_before_decision_after_processing(self):
        sim = Simulator.from_topology(self.graph(), lambda _: WAIT, processing=5)
        sim.policy = ShortestPathReference(lambda: sim.topology)
        sim.generate("M", "A", "B", deadline=20)
        sim.schedule_topology(self.graph(False), at=5)
        sim.run(20)
        self.assertEqual(sim.messages["M"].attempts, 0)

    def test_snapshot_validation_and_repeatability(self):
        logs = []
        for _ in range(2):
            sim = self.sim()
            snapshots = [TopologySnapshot(0, self.graph()), TopologySnapshot(5, self.graph(False))]
            schedule_snapshots(sim, snapshots)
            sim.generate("M", "A", "B")
            sim.run(100)
            logs.append(sim.log)
        self.assertEqual(*logs)
        sim = self.sim()
        with self.assertRaises(ValueError):
            schedule_snapshots(sim, [TopologySnapshot(0, self.graph()), TopologySnapshot(0, self.graph())])
        self.assertEqual(sim.events, [])
        with self.assertRaises(ValueError):
            sim.schedule_topology(Topology(["A"], {}), at=0)
