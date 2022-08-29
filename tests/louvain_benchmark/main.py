import random
import igraph as ig
import networkx as nx
import leidenalg as la
from collections import defaultdict
from community import community_louvain
import networkx.algorithms.community as nx_comm


G = nx.les_miserables_graph()
nx.write_gml(G, "les_miserables.gml")
H = ig.load("les_miserables.gml")
random.seed(123)

def nx_community():
    partition = nx_comm.louvain_communities(G, seed=123)
    return partition


def taynaud_community():
    partition = community_louvain.best_partition(G, random_state=123)
    return partition


def igraph_community():
    partition = H.community_multilevel()
    return partition


def leidenalg_community():
    partition = la.find_partition(H, la.ModularityVertexPartition, seed=123)
    return partition


def print_igraph_partition(partition):
    for c in partition:
        community = []
        for node in c:
            community.append(H.vs[node]["label"])
        print(sorted(community))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    partition = nx_community()
    for c in partition:
        print(sorted(c))

    print("***")

    partition = taynaud_community()
    inverted_index = defaultdict(set)
    for k, v in partition.items():
        inverted_index[v].add(k)
    for c in inverted_index.values():
        print(sorted(c))

    print("***")
    partition = igraph_community()
    print_igraph_partition(partition)


    print("***")
    partition = leidenalg_community()
    print_igraph_partition(partition)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
