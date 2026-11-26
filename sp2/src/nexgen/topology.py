"""
Topology module holds the underlying representation for a interconnected network
Undirected graph of nodes and edges. Which is a dictionary of nodes and their connections.
Nodes are expected to be either set by the user or gathered from information in the training data.
Edges are expected to be set by the user or established based on the distance between positions of nodes in the network.
(This will be obtained from the training data if not set by the user.)
Key attributes, nodes and edges.
Nodes themselves will be a data structure holding the node's position and any other relevant information.

Topology should be self handling, by this I mean it should handle its own clean up automatically
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


@dataclass
class Topology:
    _nodes: dict
    _edges: dict # Adjecency list representation
    _distances: dict # Keys are sorted device IDs joined by "->"
                     # values are the distances between the nodes    
    _max_connectivity_range: float 


    """ add_node
    Creates a new node object and adds it to the nodes dictionary.
    Takes in a device_id, position, and transmission_loss_prob as arguments.
    returns True if the node was added successfully, 
    False if the node already exists in the nodes dictionary
    or if some other error occurred while adding the node.
    """
    def add_node(self, device_id, position, transmission_loss_prob) -> bool:
        if "->" in device_id:
            return False  # Reserved edge key separator
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
    as well as another secret function distance_key to generate the standardized unique id.
    """
    @staticmethod
    def _l2(d1, d2) -> float:
        return round(math.dist(d1, d2), 2)

    @staticmethod
    def _distance_key(node_key1, node_key2) -> str:
        return "->".join(sorted((node_key1, node_key2)))

    def update_distance(self, node_key1, node_key2) -> bool:
        n1 = self._nodes[node_key1]
        n2 = self._nodes[node_key2]
        key = self._distance_key(node_key1, node_key2)

        if key not in self._distances and (node_key2 not in self._edges[node_key1]):
            return False # no connection
        
        self._distances[key] = self._l2(n1.get_position, n2.get_position)
        return True


    def get_node(self, node_key) -> Device:
        return self._node[node_key]


    def get_edges(self) -> dict: # This'll return the entire edge dictionary
        return self._edges


    def get_distance(self, node_key1, node_key2) -> float:
        key = self._distance_key(node_key1, node_key2)
        if key not in self._distances:
            return -707 # Flag for no distance found. Magic numbers are BAD.
                        # If this ever happens. I need to code in a health check.

        return self._distances[key] 


    #############################################################################################
    # Tricky because nodes should never be removed from the topology even if theyre out of range#
    # def remove_node(node_key) -> bool:
    #     pass
    #############################################################################################


    def remove_edge(self, node_key1, node_key2) -> bool:
        edges = self._edges
        if node_key2 not in edges[node_key1] or node_key1 not in edges[node_key2]:
            return False # Edge not found in one or the other or both

        edges[node_key1].pop(node_key2)
        edges[node_key2].pop(node_key1)
        return True

    
    # Since distances arent actually going to need to be removed... comment this out
    # def remove_distance(self, node_key1, node_key2) -> bool:
    #     key = self._distance_key(node_key1, node_key2)
    #     if key not in self._distances:
    #         return False # No distance found

    #     self._distances.pop(key)
    #     return True


    """
    This'll return a list of all the neighbors of a node
    """
    def neighbors(self, node_key) -> list: 
        if node_key not in self._edges:
            return [-1] # Kind of a flag for saying there is no node here. Which should never happen.

        return self._edges[node_key]

    """
    This'll return True if there is an edge between the two nodes, False otherwise
    """
    def has_edge(self, node_key1, node_key2) -> bool: 
        if node_key2 not in self._edges[node_key1]:
            return False
        else:
            return True


    # Also tricky, relies on us knowing what the shape of the data is, and thus will rely
    # on our adapter classes to distill the data into a usable data
    def apply_positions(self, timestamp, positions) -> bool:
        pass
