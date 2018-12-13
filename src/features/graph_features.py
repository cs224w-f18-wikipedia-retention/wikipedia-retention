import snap

G = snap.LoadEdgeList(snap.PUNGraph, "data/processed/user-network-v3.csv", 0, 1, "\t")

print G.GetNodes()
print G.GetEdges()

base = "data/interim/rolesense/"

def write(vec, filename):
    with open(base + filename, "w") as f:
        for item in vec:
            f.write("{},{}\n".format(item, vec[item]))

PRankH = snap.TIntFltH()
snap.GetPageRank(G, PRankH)
write(PRankH, "pr.txt")

NIdHubH = snap.TIntFltH()
NIdAuthH = snap.TIntFltH()
snap.GetHits(G, NIdHubH, NIdAuthH)
write(NIdHubH, "hub.txt")
write(NIdAuthH, "auth.txt")

Nodes = snap.TIntFltH()
Edges = snap.TIntPrFltH()
snap.GetBetweennessCentr(G, Nodes, Edges, 1.0)
write(Nodes, "between.txt")

rows = []
for i, node in enumerate(G.Nodes()):
    if i % 10000 == 0:
        print "on iteration {}".format(i)
    nid = node.GetId()
    ecc = snap.GetNodeEcc(G, nid)
    clust = snap.GetNodeClustCf(G, nid)
    rows.append([nid, ecc, clust])

with open(base + "ecc_clust.txt", 'w') as f:
    for row in rows:
        f.write(",".join(map(str, row)) + "\n")

ArtNIdV = snap.TIntV()
snap.GetArtPoints(G, ArtNIdV)

with open(base + "art.txt", "w") as f:
    for NI in ArtNIdV:
        f.write("{}\n".format(NI))


# something else
G = snap.LoadEdgeList(snap.PUNGraph, "data/processed/uu_full.csv", 0, 1, "\t")
G2 = snap.LoadEdgeList(snap.PUNGraph, "data/processed/aa_full_headers.csv", 0, 1, "\t")
snap.GetClustCf(G, 10000)

G = snap.TUNGraph.New()
d = {}
counter = 0
with open("data/processed/aa_full_headers.csv", 'r') as f:
    for line in f:
        x, y, _ = map(int, line.split())
        if x not in d:
            d[x] = counter
            G.AddNode(counter)
            counter += 1
        if y not in d:
            d[y] = counter
            G.AddNode(counter)
            counter += 1
        G.AddEdge(d[x], d[y])

G.GetNodes()
G.GedEdges()
