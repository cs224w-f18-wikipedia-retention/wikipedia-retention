# run louvian algorithm on edgelist
import snap
import numpy as np
import scipy.sparse as sparse

# load graph in SNAP for efficiency
graph_file = 'user-network-v2.csv'
Graph = snap.LoadEdgeList(snap.PUNGraph, graph_file, 0, 1)

def get_adj(Graph):
    n_nodes = Graph.GetNodes()
    n_edges = Graph.GetEdges()
    node_ids = np.array(map(lambda Node: Node.GetId(), Graph.Nodes()),dtype=int)
    # dictionary of mapped node ids for A
    node_idx = np.argsort(node_ids)
    node_dict = dict(zip(node_ids,node_idx))
    node_dict_rev = dict(zip(node_idx,node_ids))
    # build mapped edge list
    edges = list(map(lambda Edge: (Edge.GetSrcNId(),Edge.GetDstNId()), Graph.Edges()))
    mapped_edges = list(map(lambda e: (node_dict[e[0]],node_dict[e[1]]),edges))
    small_edges = np.array(map(lambda e: e[0],mapped_edges))
    big_edges = np.array(map(lambda e: e[1],mapped_edges))
    rows = np.concatenate((small_edges, big_edges))
    cols = np.concatenate((big_edges,small_edges))
    data_ones = np.ones(len(rows))
    # build adjacency matrices. NOTE: using csr is super important when looping over n rows
    A = sparse.csr_matrix((data_ones,(rows,cols)),shape=(n_nodes,n_nodes))
    return A, node_dict_rev

A, node_dict_rev = get_adj(Graph)
n = A.shape[0]

# no key -> self-community
community_dict = {}
