import sys
import glob
import pickle
import logging
import networkx as nx
import casanova
import json
import networkx.algorithms.community as nx_community
import os

logger = logging.getLogger("pimmi")


def obj_dict(obj):
    return obj.__dict__


class ClusterViz:
    def __init__(self):
        self.cluster: int = -1
        self.match_quality: float = -1
        self.sample_path: str = ""
        self.images = []

    def __repr__(self):
        return 'cluster ' + str(self.cluster) + ' (' + str(len(self.images)) + ' images)'


def from_clusters_to_viz(clusters_file, viz_file):

    logger.info("Loading %s", clusters_file.name)
    clusters = {}

    reader = casanova.reader(clusters_file)

    path_pos = reader.headers.path
    cluster_id_pos = reader.headers.cluster_id
    quality_pos = reader.headers.quality
    for row in reader:
        cluster_id = row[cluster_id_pos]
        if cluster_id in clusters:
            cluster = clusters[cluster_id]
        else:
            cluster = ClusterViz()
            cluster.cluster = cluster_id
            cluster.match_quality = row[quality_pos]
            cluster.sample_path = row[path_pos]
            clusters[cluster_id] = cluster
        cluster.images.append(row[path_pos])
    reader.close()

    viz_file.write(json.dumps(list(clusters.values()),
                   default=obj_dict, indent=4))
    viz_file.close()
    logger.info("Conversion to JSON done.")


def generate_graph_from_files(file_patterns, min_nb_match_ransac):
    if file_patterns != "":
        all_files = glob.glob(file_patterns)
        for filename in all_files:
            if os.stat(filename).st_size != 0:
                with open(filename) as f:
                    reader = casanova.reader(f)
                    query_image_id = reader.headers["query_image_id"]
                    result_image_id = reader.headers["result_image_id"]
                    nb_match_ransac = reader.headers["nb_match_ransac"]

                    for row in reader:
                        if row[query_image_id] != row[result_image_id] and int(row[nb_match_ransac]) >= min_nb_match_ransac:
                            yield int(row[query_image_id]), int(row[result_image_id]), int(row[nb_match_ransac])
    else:
        reader = casanova.reader(sys.stdin)
        query_image_id = reader.headers["query_image_id"]
        result_image_id = reader.headers["result_image_id"]
        nb_match_ransac = reader.headers["nb_match_ransac"]

        for row in reader:
            if row[query_image_id] != row[result_image_id] and int(row[nb_match_ransac]) >= min_nb_match_ransac:
                yield int(row[query_image_id]), int(row[result_image_id]), int(row[nb_match_ransac])


def metrics_from_subgraph(enum, sg):
    return sum(w for u, v, w in sg.edges.data("weight")), enum, sg.degree()


def yield_communities(g, algo="components"):

    if g.is_directed():
        components = nx.weakly_connected_components
    else:
        components = nx.connected_components

    if algo == "louvain":
        largest_cc = max(components(g), key=len)

    community_counter = 0
    for component_id, component in enumerate(components(g)):

        sg = g.subgraph(component)
        if algo == "components":
            yield metrics_from_subgraph(component_id, sg)

        elif algo == "louvain":
            if component == largest_cc:
                undirected_sg = sg.to_undirected()
                communities = nx_community.louvain_communities(undirected_sg)
                for community in communities:
                    community_counter += 1
                    yield metrics_from_subgraph(community_counter, undirected_sg.subgraph(community))
            else:
                community_counter += 1
                yield metrics_from_subgraph(community_counter, sg)

    logger.info("Connected components in the graph: %d", component_id + 1)

    if algo == "louvain":
        logger.info("Louvain communities in the graph: %d", community_counter)


def generate_clusters(results_pattern, merged_meta_file, clusters_file, nb_match_ransac, algo):
    logger.info("Loading query results")

    if algo == "louvain":
        g = nx.Graph()
        for u, v, weight in generate_graph_from_files(results_pattern, nb_match_ransac):
            if g.has_edge(u, v):
                g[u][v]["weight"] += weight
            else:
                g.add_edge(u, v, weight=weight)

    elif algo == "components":
        g = nx.DiGraph()
        for u, v, weight in generate_graph_from_files(results_pattern, nb_match_ransac):
            g.add_edge(u, v, weight=weight)

    else:
        raise ValueError("'algo' must be set to 'louvain' or 'components'")

    logger.info("Graph contains {} nodes and {} edges".format(len(g), g.size()))

    with open(merged_meta_file, 'rb') as f:
        meta_json = pickle.load(f)

    writer = casanova.writer(
        clusters_file,
        ["path", "image_id", "nb_points", "degree", "cluster_id", "quality"],
    )
    for nb_matches, community_id, node_degrees in yield_communities(g, algo):
        nb_points = []
        paths = []

        for node_id, _ in node_degrees:
            meta_image = meta_json[node_id]
            nb_points.append(meta_image["nb_points"])
            paths.append(meta_image["path"])

        sum_weight = range(len(nb_points), 0, -1)
        max_theoretical_matches = 2 * \
            sum([i * j for i, j in zip(sorted(nb_points), sum_weight)])
        quality = nb_matches / max_theoretical_matches

        for node_degree, nb, path in zip(node_degrees, nb_points, paths):
            node_id, degree = node_degree
            writer.writerow([path, node_id, nb, degree, community_id, quality])
    clusters_file.close()


if __name__ == '__main__':
    merged_meta_file = "index/dataset1.IVF1024,Flat.meta"
    results_pattern = "index/dataset1.IVF1024,Flat.mining_*"
    clusters_file = "index/dataset1.IVF1024,Flat.mining.clusters.csv"

    generate_clusters(
        results_pattern,
        merged_meta_file,
        clusters_file,
        nb_match_ransac=10,
        algo="louvain",
        edge_collapse="mean"
    )
