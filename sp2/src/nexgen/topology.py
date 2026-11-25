"""
Topology module holds the underlying representation for a interconnected network
Undirected graph of nodes and edges. Which is a dictionary of nodes and their connections.
Nodes are expected to be either set by the user or gathered from information in the training data.
Edges are expected to be set by the user or established based on the distance between positions of nodes in the network.
(This will be obtained from the training data if not set by the user.)
Key attributes, nodes and edges.
Nodes themselves will be a data structure holding the node's position and any other relevant information.
"""
from dataclasses import dataclass


@dataclass
class Device:
    device_id: str
    position: tuple  # x, y
    transmission_loss_prob: float

    @property
    def device(self):
        return self.device_id

    @property
    def get_position(self) -> tuple:
        return self.position

    def update_position(self, new_position) -> None:
        self.position = new_position

    @property
    def get_transmission_loss(self) -> float:
        return self.transmission_loss_prob


"""
To do:
class Topology:
    nodes: dict of Device objects
    edges: dict of edges (key: device, value: list of connected devices) 
    distances: dict of distances between devices [Edge] = distance d
    methods:
    constructor 
    add_node
    add_edge
    remove_node
    remove_edge
    update distance
    # For when a node falls out of the network or we want to remove a connection between two nodes
    # ...due to the distance between the nodes being too great or some other reason.
    # Maybe we can define nodes to have a failure rate and if a node fails it is removed from the network 
    # and all edges connected to it are also removed.

"""