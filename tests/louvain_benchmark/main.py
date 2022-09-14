import random
import timeit
import igraph as ig
import pandas as pd
import networkx as nx
import leidenalg as la
import seaborn as sns
import matplotlib.pyplot as plt
from community import community_louvain
import networkx.algorithms.community as nx_comm


random.seed(123)
n_groups = 4
n_vertices = [100, 1000, 10000]


def networkx_louvain(G):
    partition = nx_comm.louvain_communities(G, seed=123)
    return partition


def taynaud_louvain(G):
    partition = community_louvain.best_partition(G, random_state=123)
    return partition


def igraph_louvain(H):
    partition = H.community_multilevel()
    return partition


def leidenalg_modularity(H):
    partition = la.find_partition(H, la.ModularityVertexPartition, seed=123)
    return partition


def print_igraph_partition(partition):
    for c in partition:
        community = []
        for node in c:
            community.append(H.vs[node]["label"])
        print(sorted(community))

if __name__ == '__main__':
    results = []

    for k in n_vertices:
        G = nx.planted_partition_graph(n_groups, k, 0.03, 0.003)
        edges = G.number_of_edges()
        nx.write_gml(G, "planted_partition.gml", stringizer=str)
        H = ig.load("planted_partition.gml")

        for implementation, graph in zip(
                ["networkx_louvain", "taynaud_louvain", "leidenalg_modularity", "igraph_louvain"],
                ["G", "G", "H", "H"]
        ):
            def append_results(number, time_taken):
                results.append(
                    {"nodes": n_groups*k, "edges": edges,  "version": implementation, "time": time_taken/number}
                )

            print(implementation)
            t = timeit.Timer(
                '{}({})'.format(implementation, graph),
                setup="from __main__ import {}, {}; gc.enable()".format(implementation, graph)
            )
            for i in range(3):
                t.autorange(callback=append_results)
    pd.DataFrame(results).to_csv("louvain_results.csv")
    ax = sns.barplot(x="nodes", y="time", hue="version", ci="sd", data=pd.DataFrame(results))
    ax.set_yscale("log")
    plt.savefig("louvain_results.jpg")
