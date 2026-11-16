import unittest
from nexgen_sim import Generation, Link, Simulator, Stream, Topology, WAIT, Workload, periodic, poisson


class TrafficTests(unittest.TestCase):
    def graph(self):
        return Topology(["A", "B"], {("A", "B"): Link(10)})

    def test_replay_across_policies_and_json(self):
        workload = periodic("s", "A", "B", interval=10, lifetime=50, start=0, stop=30)
        loaded = Workload.from_json(workload.to_json())
        self.assertEqual(workload, loaded)
        generations = []
        for policy in (lambda _: "B", lambda _: WAIT):
            sim = Simulator.from_topology(self.graph(), policy)
            loaded.replay(sim)
            sim.run(100)
            generations.append([(m.id, m.source, m.destination, m.generated_at, m.deadline)
                                for m in sim.messages.values()])
            self.assertEqual(sim.accounting()["generated"], 3)
            self.assertEqual(sim.accounting()["pending"], 0)
        self.assertEqual(*generations)
        self.assertEqual([r.deadline for r in loaded.records], [50, 60, 70])

    def test_replay_rejection_is_atomic(self):
        sim = Simulator.from_topology(self.graph(), lambda _: "B")
        invalid = Workload((Generation("a", "A", "B", 0, 50), Generation("b", "A", "missing", 1, 50)))
        with self.assertRaises(ValueError):
            invalid.replay(sim)
        self.assertEqual(sim.events, [])
        valid = Workload((Generation("a", "A", "B", 0, 50),))
        valid.replay(sim)
        before = list(sim.events)
        with self.assertRaises(ValueError):
            valid.replay(sim)
        self.assertEqual(sim.events, before)

    def test_poisson_stream_independence_and_lifetimes(self):
        a = Stream("a", "A", "B", 100, 50, 10, 1_000_000)
        b = Stream("b", "A", "B", 200, 75, 10, 1_000_000)
        options = dict(seed=42, topology=self.graph())
        both = poisson([a, b], **options)
        self.assertEqual(both, poisson([b, a], **options))
        alone = poisson([a], **options)
        self.assertTrue(set(alone.records).issubset(both.records))
        self.assertNotEqual(both, poisson([a, b], seed=43, topology=self.graph()))
        for r in both.records:
            self.assertTrue(10 < r.at < 1_000_000)
            self.assertEqual(r.deadline - r.at, 50 if r.id.startswith('["a",') else 75)
        self.assertTrue(200 < len(both.records) < 400)

    def test_quantized_arrivals_and_rate(self):
        low = Stream("s", "A", "B", 100, 50, 0, 1_000_000)
        high = Stream("s", "A", "B", 1000, 50, 0, 1_000_000)
        opts = dict(seed=7, topology=self.graph())
        self.assertGreater(len(poisson([high], **opts).records), len(poisson([low], **opts).records))
        dense = poisson([Stream("dense", "A", "B", 1e7, 50, 0, 10)], **opts)
        self.assertGreater(len(dense.records), len({r.at for r in dense.records}))
        self.assertTrue(all(type(r.at) is int and r.at < 10 for r in dense.records))

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(poisson([Stream("s", "A", "B", 0, 5, 0, 10)], seed=0, topology=self.graph()).records, ())
        for rate in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                Stream("s", "A", "B", rate, 5, 0, 10)
        s = Stream("s", "A", "B", 1, 5, 0, 10)
        with self.assertRaises(ValueError):
            poisson([s, s], seed=0, topology=self.graph())
        with self.assertRaises(ValueError):
            periodic("s", "A", "B", interval=0, lifetime=5, start=0, stop=10)
        with self.assertRaises(ValueError):
            Generation("m", "A", "B", 5, 5)
