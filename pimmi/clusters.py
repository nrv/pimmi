import sys
import glob
import pickle
import logging
import networkx as nx
import casanova
import json

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
    logger.info("Loading %s", clusters_file)
    clusters = {}
    with open(clusters_file) as infile:
        reader = casanova.reader(infile)
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
    logger.info("Writing %s", viz_file)
    with open(viz_file, 'w') as outfile:
        outfile.write(json.dumps(list(clusters.values()), default=obj_dict, indent=4))


def generate_graph_from_files(file_patterns, min_nb_match_ransac):
    all_files = glob.glob(file_patterns)

    for filename in all_files:
        with open(filename) as f:
            reader = casanova.reader(f)
            query_image_id = reader.headers["query_image_id"]
            result_image_id = reader.headers["result_image_id"]
            nb_match_ransac = reader.headers["nb_match_ransac"]

            for row in reader:
                if row[query_image_id] != row[result_image_id] and int(row[nb_match_ransac]) >= min_nb_match_ransac:
                    yield int(row[query_image_id]), int(row[result_image_id]), int(row[nb_match_ransac])


def metrics_from_subgraph(enum, sg):
    return sum(w for u, v, w in sg.edges.data("weight")), enum, sg.degree()


def yield_communities(g, algo="components", edge_collapse="mean"):
    comp = nx.weakly_connected_components(g)

    community_counter = 0
    for component_id, component in enumerate(comp):

        sg = g.subgraph(component)
        if algo == "components":
            yield metrics_from_subgraph(component_id, sg)

        elif algo == "louvain":
            sg.to_undirected(combine_edges=edge_collapse)
            sub_sub_graphs = sg.community_multilevel().subgraphs()
            for community_id, ssg in enumerate(sub_sub_graphs):
                community_counter += 1
                yield metrics_from_subgraph(community_id, ssg)

    logger.info("Connected components in the graph: %d", component_id + 1)

    if algo == "louvain":
        logger.info("Louvain communities in the graph: %d", community_counter)


def generate_clusters(results_pattern, merged_meta_file, clusters_file, nb_match_ransac, algo, edge_collapse):
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

    f = open(clusters_file, 'w') if clusters_file else sys.stdout

    writer = casanova.writer(f, ["path", "image_id", "nb_points", "degree", "cluster_id", "quality"])
    for nb_matches, community_id, node_degrees in yield_communities(g, algo, edge_collapse):
        nb_points = []
        paths = []

        for node_id, _ in node_degrees:
            meta_image = meta_json[node_id]
            nb_points.append(meta_image["nb_points"])
            paths.append(meta_image["path"])

        sum_weight = range(len(nb_points), 0, -1)
        max_theoretical_matches = 2 * sum([i * j for i, j in zip(sorted(nb_points), sum_weight)])
        quality = nb_matches / max_theoretical_matches

        for node_degree, nb, path in zip(node_degrees, nb_points, paths):
            node_id, degree = node_degree
            writer.writerow([path, node_id, nb, degree, community_id, quality])

    f.close()


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
