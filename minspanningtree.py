"""
Minimum Spanning Tree Implementation
Author: Riley McKenzie
"""

import time


def createAdjMatrix(filename):
    """
    Reads a file containing an adjacency matrix and returns it as a list of lists.
    Each row in the file represents a row in the adjacency matrix.
    """
    with open(filename, 'r') as f:
        return [list(map(int, line.split())) for line in f]


def prim(W):
    """
    Implements Prim's algorithm to find the Minimum Spanning Tree (MST).
    
    Args:
        W (list of list of int): Adjacency matrix where W[i][j] is the weight of edge (i, j).
    
    Returns:
        list of tuples: Edges in the MST in the form (i, j, weight), with i < j.
    """
    import heapq

    num_vertices = len(W)
    visited = set()
    mst_edges = []
    min_heap = [(0, 0, 0)]  # (weight, vertex_from, vertex_to)

    while len(visited) < num_vertices:
        weight, u, v = heapq.heappop(min_heap)
        if v in visited:
            continue
        
        visited.add(v)
        if u != v:
            mst_edges.append((min(u, v), max(u, v), weight))
        
        for neighbor in range(num_vertices):
            if neighbor not in visited and W[v][neighbor] > 0:
                heapq.heappush(min_heap, (W[v][neighbor], v, neighbor))
    
    return mst_edges


def kruskal(W):
    """
    Implements Kruskal's algorithm to find the Minimum Spanning Tree (MST).
    
    Args:
        W (list of list of int): Adjacency matrix where W[i][j] is the weight of edge (i, j).
    
    Returns:
        list of tuples: Edges in the MST in the form (i, j, weight), with i < j.
    """
    num_vertices = len(W)
    edges = []

    # Convert adjacency matrix to edge list
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if W[i][j] > 0:
                edges.append((W[i][j], i, j))  # (weight, vertex_1, vertex_2)

    edges.sort()  # Sort edges by weight

    parent = list(range(num_vertices))
    rank = [0] * num_vertices
    mst_edges = []

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(v1, v2):
        root1, root2 = find(v1), find(v2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, weight))
            if len(mst_edges) == num_vertices - 1:
                break
    
    return mst_edges


if __name__ == '__main__':
    g = createAdjMatrix("graph_verts10.txt")

    # Run Prim's algorithm
    start_time = time.time()
    res_prim = prim(g)
    elapsed_time_prim = time.time() - start_time
    print(f"Prim's runtime: {elapsed_time_prim:.2f}")

    # Run Kruskal's algorithm
    start_time = time.time()
    res_kruskal = kruskal(g)
    elapsed_time_kruskal = time.time() - start_time
    print(f"Kruskal's runtime: {elapsed_time_kruskal:.2f}")

    # Check that sum of edge weights are the same for both algorithms
    cost_prim = sum([e[2] for e in res_prim])
    print("MST cost w/ Prim: ", cost_prim)
    cost_kruskal = sum([e[2] for e in res_kruskal])
    print("MST cost w/ Kruskal: ", cost_kruskal)
    assert cost_prim == cost_kruskal
