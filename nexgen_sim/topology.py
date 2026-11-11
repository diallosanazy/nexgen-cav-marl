"""Static graphs and their versioned JSON representation."""

import json
from types import MappingProxyType

from .engine import Link


def metadata(value):
    """Copy JSON data; never retain caller-owned mutable metadata."""
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("metadata must be an object with string keys")
    return json.loads(json.dumps(value, allow_nan=False))


class Topology:
    def __init__(self, nodes, links, *, node_attributes=None, edge_attributes=None):
        ids = list(nodes)
        if any(not isinstance(n, str) for n in ids) or len(set(ids)) != len(ids):
            raise ValueError("node IDs must be unique strings")
        self.nodes = tuple(sorted(ids))
        node_set = set(ids)
        for edge, link in links.items():
            if (not isinstance(edge, tuple) or len(edge) != 2 or
                    any(n not in node_set for n in edge) or edge[0] == edge[1]):
                raise ValueError("edges must connect distinct known nodes")
            if not isinstance(link, Link):
                raise ValueError("edges must contain Link values")
        self.links = MappingProxyType(dict(sorted(links.items())))
        outgoing = {n: [] for n in self.nodes}
        for a, b in self.links:
            outgoing[a].append(b)
        self._neighbors = {n: tuple(v) for n, v in outgoing.items()}
        node_attributes = node_attributes or {}
        edge_attributes = edge_attributes or {}
        if set(node_attributes) - node_set or set(edge_attributes) - set(links):
            raise ValueError("attributes reference unknown nodes or edges")
        self._node_attributes = {n: metadata(node_attributes.get(n, {})) for n in self.nodes}
        self._edge_attributes = {e: metadata(edge_attributes.get(e, {})) for e in self.links}

    def neighbors(self, node):
        return self._neighbors[node]

    def node_attributes(self, node):
        return metadata(self._node_attributes[node])

    def edge_attributes(self, source, target):
        return metadata(self._edge_attributes[source, target])

    def to_json(self):
        return json.dumps({"version": 1,
                           "nodes": [{"id": n, "metadata": self._node_attributes[n]} for n in self.nodes],
                           "edges": [{"source": a, "target": b,
                                      "transmission_us": link.transmission,
                                      "propagation_us": link.propagation,
                                      "metadata": self._edge_attributes[a, b]}
                                     for (a, b), link in self.links.items()]},
                          sort_keys=True, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        try:
            data = json.loads(text)
            if data["version"] != 1:
                raise ValueError("unsupported topology version")
            nodes = [n["id"] for n in data["nodes"]]
            links, attrs = {}, {}
            for e in data["edges"]:
                key = (e["source"], e["target"])
                if key in links:
                    raise ValueError("duplicate edge")
                links[key] = Link(e["transmission_us"], e["propagation_us"])
                attrs[key] = e.get("metadata", {})
            return cls(nodes, links,
                       node_attributes={n["id"]: n.get("metadata", {}) for n in data["nodes"]},
                       edge_attributes=attrs)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("invalid topology JSON") from exc
