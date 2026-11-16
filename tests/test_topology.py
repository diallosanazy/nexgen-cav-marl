import json
import unittest
from nexgen_sim import Link, Simulator, Topology, WAIT


class TopologyTests(unittest.TestCase):
    def graph(self):
        return Topology(["taxi-42", "destination", "isolated", "middle"],
                        {("taxi-42", "middle"): Link(10, 2),
                         ("middle", "destination"): Link(10, 2)},
                        node_attributes={"taxi-42": {"position": [1, 2]}},
                        edge_attributes={("taxi-42", "middle"): {"loss_probability": 1.0}})

    def test_roundtrip_trace_and_metadata(self):
        graph = self.graph()
        loaded = Topology.from_json(graph.to_json())
        self.assertEqual(graph.to_json(), loaded.to_json())
        logs = []
        for topology in (graph, loaded):
            sim = Simulator.from_topology(topology, lambda o: WAIT if o.time == 0 else o.neighbors[0])
            sim.generate("M", "taxi-42", "destination")
            sim.run(50)
            self.assertEqual(sim.accounting()["delivered"], 1)  # loss metadata is inactive
            logs.append(sim.log)
        self.assertEqual(*logs)
        attrs = loaded.node_attributes("taxi-42")
        attrs["position"].append(3)
        self.assertEqual(loaded.node_attributes("taxi-42"), {"position": [1, 2]})

    def test_neighbors_and_isolates(self):
        graph = Topology(["b", "a", "c", "isolate"],
                         {("a", "c"): Link(1), ("a", "b"): Link(1)})
        self.assertEqual(graph.neighbors("a"), ("b", "c"))
        self.assertEqual(graph.neighbors("isolate"), ())
        with self.assertRaises(TypeError):
            graph.links["c", "b"] = Link(1)

    def test_invalid_graphs(self):
        for nodes, links in [(["a", "a"], {}), ([1], {}),
                             (["a"], {("a", "b"): Link(1)}),
                             (["a"], {("a", "a"): Link(1)}),
                             (["a", "b"], {("a", "b"): 5})]:
            with self.subTest(nodes=nodes, links=links), self.assertRaises(ValueError):
                Topology(nodes, links)
        data = json.loads(self.graph().to_json())
        data["edges"].append(data["edges"][0])
        with self.assertRaises(ValueError):
            Topology.from_json(json.dumps(data))
        with self.assertRaises(ValueError):
            Topology.from_json('{}')
