import glob
import pickle
import logging
import igraph as ig
import casanova


# TODO parameters from command line
logger = logging.getLogger("gener")


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


def generate_clusters(results_pattern, merged_meta_file, viz_data_file, nb_match_ransac):

    logger.info("Loading query results")

    g = ig.Graph.TupleList(generate_graph_from_files(results_pattern, nb_match_ransac), directed=True, weights=True)

    logger.info("Number of vertices in the graph: %d", g.vcount())
    logger.info("Number of edges in the graph: %d", g.ecount())
    logger.info("Is the graph directed: %d", g.is_directed())
    logger.info("Maximum degree in the graph: %d", g.maxdegree())

    comp = g.components(mode="weak")
    logger.info("Connected components in the graph: %d", len(comp))

    with open(merged_meta_file, 'rb') as f:
        meta_json = pickle.load(f)

    sub_graphs = comp.subgraphs()
    with open(viz_data_file, "w") as f:
        writer = casanova.writer(f, ["path", "image_id", "nb_points", "degree", "cluster_id", "quality"])
        for cluster_id, sg in enumerate(sub_graphs):
            nb_points = []
            paths = []
            for image_id in sg.vs["name"]:
                meta_image = meta_json[image_id]
                nb_points.append(meta_image["nb_points"])
                paths.append(meta_image["path"])
            sum_weight = range(len(nb_points), 0, -1)
            max_theoretical_matches = 2 * sum([i * j for i, j in zip(sorted(nb_points), sum_weight)])
            nb_matches = sum(sg.es["weight"])
            quality = nb_matches / max_theoretical_matches

            for node, path, nb in zip(sg.vs, paths, nb_points):
                writer.writerow([path, node["name"], nb, node.degree(), cluster_id, quality])


if __name__ == '__main__':
    merged_meta_file = "index/dataset1.ivf1024.meta"
    results_pattern = "index/dataset1.ivf1024.mining_*"
    viz_data_file = "index/dataset1.ivf1024.mining.clusters.json"
    generate_clusters()
