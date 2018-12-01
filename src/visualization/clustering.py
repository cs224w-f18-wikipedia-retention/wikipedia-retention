import snap
import numpy as np
import pandas as pd

path = "data/processed/enwiki-projection-user-dev.csv"
print("loading {}".format(path))
G = snap.LoadEdgeList(snap.PUNGraph, path, 0, 1, '\t')


print("nodes", G.GetNodes())
print("edges", G.GetEdges())

print("average clustering coeff", snap.GetClustCf(G))

admin_path = "data/processed/admin_mapping.csv"
admin_df = pd.read_csv(admin_path, header=None, names=['user_id', 'username'])

def calculate_clustering_coeff(G, nodes):
    coeffs = []
    for node in nodes:
        if not G.IsNode(node):
            continue
        coeffs.append(snap.GetNodeClustCf(G, node))
    avg = np.average(coeffs)
    std = np.std(coeffs)
    n = len(coeffs)
    err = 1.96*std/np.sqrt(n)
    print(
        "avg: {:0.3f}\tstd: {:0.3f}\tsamples: {}\terr: {:0.3f}"
        .format(avg, std, n, err)
    )

calculate_clustering_coeff(G, admin_df.user_id.values)
calculate_clustering_coeff(G, [G.GetRndNId() for _ in range(5000)])