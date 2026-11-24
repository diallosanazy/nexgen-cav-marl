"""
Topology module holds the underlying representation for a interconnected network
Undirected graph of nodes and edges. Which is a dictionary of nodes and their connections.
Nodes are expected to be either set by the user or gathered from information in the training data.
Edges are expected to be set by the user or established based on the distance between positions of nodes in the network.
(This will be obtained from the training data if not set by the user.)
Key attributes, nodes and edges.
Nodes themselves will be a data structure holding the node's position and any other relevant information.
"""


"""
To do:
class Node:
    position x, y
    failure_rate
    connectivity # How strong is the connection to other nodes? This can be a function of distance and failure rate.


class Topology:
    nodes: dict of Node objects
    edges: dict of edges (key: node, value: list of connected nodes) 
    # I wanted to store also the distance between the nodes in a given edge relationship but I feel like in the real world thats unnecessary?

    methods:
    constructor 
    add_node
    add_edge
    remove_node
    remove_edge
    # For when a node falls out of the network or we want to remove a connection between two nodes
    # ...due to the distance between the nodes being too great or some other reason.
    # Maybe we can define nodes to have a failure rate and if a node fails it is removed from the network 
    # and all edges connected to it are also removed.

"""