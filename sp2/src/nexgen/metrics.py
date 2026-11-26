"""
To consolidate all the metrics for the research project
* Packet counts
* Packet delivery ratio
* Throughput
* End-to-end latency
* Deadline expirations
* Drops
* Forwarding and routing overhead
* Computational cost
* ~~Topology measurements (ie link counts, degree, connected components, links added/removed per frame
                         how many traffic pairs were direct, multi-hop or unreachable)~~
* Age of Information (AoI)
* Deadline miss rate and LVR
"""

def topology_metrics(edges) -> dict:
    degree_by_device = {}
    total_degree = 0

    for node_key in edges:
        degree = len(edges[node_key])
        degree_by_device[node_key] = degree
        total_degree += degree

    unvisited = set(edges)
    connected_components = 0

    while unvisited:
        # Start with one device and walk through everything it can reach.
        stack = [unvisited.pop()]
        connected_components += 1

        while stack:
            node_key = stack.pop()

            for neighbor in edges[node_key]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)

    device_count = len(edges)
    mean_degree = None
    if device_count > 0:
        mean_degree = total_degree / device_count

    return {
        "device_count": device_count,
        "link_count": total_degree // 2,
        "degree_by_device": degree_by_device,
        "mean_degree": mean_degree,
        "connected_components": connected_components,
    }