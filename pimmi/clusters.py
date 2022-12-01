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


def yield_communities(g, algo="components"):
    comp = g.components(mode="weak")
    logger.info("Connected components in the graph: %d", len(comp))
    sub_graphs = comp.subgraphs()

    for component_id, sg in enumerate(sub_graphs):
        if algo == "components":
            nb_matches = sum(sg.es["weight"])
            yield nb_matches, component_id, sg.vs["name"], sg.vs.degree()

        elif algo == "louvain":
            sg.to_undirected(combine_edges="sum")
            for community_id, community in enumerate(sg.community_multilevel()):
                nb_matches = sum(sg.es.select(_within=community)["weight"])
                degrees = sg.vs.select(community).degree()
                yield nb_matches, community_id, community, degrees

        else:
            raise ValueError("'algo' must be set to 'louvain' or 'components'")


def generate_clusters(results_pattern, merged_meta_file, viz_data_file, nb_match_ransac, algo):
    logger.info("Loading query results")

    g = ig.Graph.TupleList(generate_graph_from_files(results_pattern, nb_match_ransac), directed=True, weights=True)

    logger.info("Number of vertices in the graph: %d", g.vcount())
    logger.info("Number of edges in the graph: %d", g.ecount())
    logger.info("Is the graph directed: %d", g.is_directed())
    logger.info("Maximum degree in the graph: %d", g.maxdegree())

    with open(merged_meta_file, 'rb') as f:
        meta_json = pickle.load(f)

    with open(viz_data_file, "w") as f:
        writer = casanova.writer(f, ["path", "image_id", "nb_points", "degree", "cluster_id", "quality"])
        for nb_matches, community_id, node_ids, degrees in yield_communities(g, algo):
            nb_points = []
            paths = []

            for node_id in node_ids:
                meta_image = meta_json[node_id]
                nb_points.append(meta_image["nb_points"])
                paths.append(meta_image["path"])

            sum_weight = range(len(nb_points), 0, -1)
            max_theoretical_matches = 2 * sum([i * j for i, j in zip(sorted(nb_points), sum_weight)])
            quality = nb_matches / max_theoretical_matches

            for node_id, nb, path, degree in zip(node_ids, nb_points, paths, degrees):
                writer.writerow([path, node_id, nb, degree, community_id, quality])


if __name__ == '__main__':
    merged_meta_file = "index/dataset1.IVF1024,Flat.meta"
    results_pattern = "index/dataset1.IVF1024,Flat.mining_*"
    viz_data_file = "index/dataset1.IVF1024,Flat.mining.clusters.json"
    generate_clusters(results_pattern, merged_meta_file, viz_data_file, nb_match_ransac=10, algo="louvain")
