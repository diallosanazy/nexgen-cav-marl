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
import math


@dataclass
class Device:
    device_id: str
    position: tuple  # x, y
    transmission_loss_prob: float

    @property
    def get_deviceid(self):
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
@dataclass
class Topology:
    _nodes: dict
    _edges: dict # Adjecency list representation
    _distances: dict # Keys are strings of the form "node1-node2" 
                     # values are the distances between the nodes    


    """ add_node
    Creates a new node object and adds it to the nodes dictionary.
    Takes in a device_id, position, and transmission_loss_prob as arguments.
    returns True if the node was added successfully, 
    False if the node already exists in the nodes dictionary
    or if some other error occurred while adding the node.
    """
    def add_node(self, device_id, position, transmission_loss_prob) -> bool:
        if device_id in self._nodes:
            return False  # Node already exists
        new_device = Device(device_id, position, transmission_loss_prob)
        self._nodes[device_id] = new_device
        self._edges[device_id] = []
        return True


    """ add_edge
    Creates an edge between two devices in the network. Takes in two device_ids as arguments.
    returns True if the edge was added successfully,
    False if the edge already exists in the edges dictionary
    or if some other error occurred while adding the edge.
    It should naturally check if the edge id is flipped and if so, 
    it should not add the edge again.
    As well as if a device is in the network.
    """
    def add_edge(self, node_key1, node_key2) -> bool:
        if node_key1 not in self._nodes or node_key2 not in self._nodes:
            return False 
        elif node_key1 == node_key2:
            return False 
        elif node_key1 in self._edges and node_key2 in self._edges[node_key1]:
            return False  

        self._edges[node_key1].append(node_key2)
        self._edges[node_key2].append(node_key1) # Reflect that theyre connected neighbors
        self.update_distance(node_key1, node_key2) # log their distance
        return True
        

    """ update_distance
    Creates a unique key using the nodes if it doesn't already exist
    Combining the role of creating an entry of distance for the first time
    or updating a preexisting distance value
    uses a secret l2 aux function to determine the distances based on x-y positioning of devices
    """
    @staticmethod
    def _l2(d1, d2) -> float:
        return math.dist(d1, d2)

    def update_distance(self, node_key1, node_key2) -> bool:
        n1 = self._nodes[node_key1]
        n2 = self._nodes[node_key2]
        uid = n1.get_deviceid() + n2.get_deviceid()
        ruid = n2.get_deviceid() + n1.get_deviceid()

        if uid not in self._distances or ruid not in self._distances:
            self._distances[uid] = self._l2(n1.get_position(), n2.get_position())
        elif node_key2 not in self.neighbors(node_key1):
            return False # Not even connected.
        else:
            if uid in self._distances:
                self._distances[uid] = self._l2(n1.get_position(), n2.get_position())
            elif ruid in self._distances:
                self._distances[ruid] = self._l2(n1.get_position(), n2.get_position())
        
        return True


    def get_node(node_key) -> Device:
        pass


    def get_edges() -> dict: # This'll return the entire edge dictionary
        pass


    def get_distance(node_key1, node_key2) -> float:
        pass


    def remove_node(node_key) -> bool:
        pass


    def remove_edge(node_key1, node_key2) -> bool:
        pass

    
    def remove_distance(node_key1, node_key2) -> bool:
        pass


    """
    This'll return a list of all the neighbors of a node
    """
    def neighbors(node_key) -> list: 
        pass

    """
    This'll return True if there is an edge between the two nodes, False otherwise
    """
    def has_edge(node_key1, node_key2) -> bool: 
        pass


    def apply_positions(timestamp, positions) -> bool:
        pass
