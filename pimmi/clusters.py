import glob
import pickle
import logging
import pandas as pd
import numpy as np
import igraph as ig
import casanova

import pimmi.pimmi_parameters as constants


# TODO parameters from command line
logger = logging.getLogger("gener")


def from_cluster_2col_table_to_list(data, keycol, valcol):
    keys, values = data.sort_values(keycol).values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    return pd.DataFrame({keycol: ukeys, valcol: [list(a) for a in arrays]})


def generate_graph_from_files(file_patterns, min_nb_match_ransac=10):
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


def generate_clusters(results_pattern, merged_meta_file, viz_data_file):

    logger.info("Loading query results")

    g = ig.Graph.TupleList(generate_graph_from_files(results_pattern), directed=True, weights=True)

    logger.info("Number of vertices in the graph: %d", g.vcount())
    logger.info("Number of edges in the graph: %d", g.ecount())
    logger.info("Is the graph directed: %d", g.is_directed())
    logger.info("Maximum degree in the graph: %d", g.maxdegree())

    comp = g.components(mode="weak")
    logger.info("Connected components in the graph: %d", len(comp))
    clusters = pd.DataFrame(list(zip(g.vs["name"], comp.membership)), columns=['image_id', 'cluster'])
    logger.info("Images in clusters : %d", len(clusters))

    with open(merged_meta_file, 'rb') as f:
        meta_json = pickle.load(f)

    index_meta_df = pd.DataFrame(meta_json.items(), columns=["tmp_id", "tmp_stuff"])
    unnested = pd.json_normalize(index_meta_df["tmp_stuff"])
    all_meta = index_meta_df.join(unnested).drop(columns=["tmp_id", "tmp_stuff"])

    clusters_with_meta = pd.merge(clusters, all_meta, how="left", left_on="image_id", right_on="image_id")
    # print(clusters_with_meta.head(10))

    # summary_clusters = clusters_with_meta.groupby("cluster").agg({prm.mex_nbSeen: ['sum', 'count'],
    #                                                               prm.mex_relative_path: ['first']})
    # summary_clusters.columns = ['nb_seen', 'nb_images', 'sample_path']
    # summary_clusters = summary_clusters.reset_index()
    # logger.info("%d distinct clusters", len(summary_clusters))
    # summary_clusters = summary_clusters.sort_values(by=['nb_seen', 'nb_images'], ascending=False)

    summary_clusters = clusters_with_meta.groupby("cluster").agg({constants.dff_image_path: ['first', 'count']})
    summary_clusters.columns = ['sample_path', 'nb_images']
    summary_clusters = summary_clusters.reset_index()
    logger.info("%d distinct clusters", len(summary_clusters))

    sub_graphs = comp.subgraphs()
    for cluster_id, sg in enumerate(sub_graphs):
        # cluster_id = 1
        # sg = sub_graphs[cluster_id]
        nb_points = sorted(clusters_with_meta[clusters_with_meta["cluster"] == cluster_id]["nb_points"])
        sum_weight = list(range(1, len(nb_points) + 1))
        sum_weight.reverse()
        max_theoretical_matches = 2 * sum([i * j for i, j in zip(nb_points, sum_weight)])
        nb_matches = sum(sg.es["weight"])
        quality = nb_matches / max_theoretical_matches
        summary_clusters.loc[cluster_id, "match_quality"] = quality
        logger.info(
            "cluster %d has %d vertices and %d edges, quality index : %f", cluster_id, sg.vcount(), sg.ecount(), quality
        )

    print(summary_clusters.head(10))

    clusters_viz = from_cluster_2col_table_to_list(
        clusters_with_meta[["cluster", constants.dff_image_path]],
        "cluster",
        "images"
    )
    clusters_viz = pd.merge(clusters_viz, summary_clusters, how="left", left_on="cluster", right_on="cluster")
    clusters_viz = clusters_viz.sort_values(by=['nb_images'], ascending=False)
    print(clusters_viz.head(10))

    # viz_data = list(clusters_viz["relativePath"])
    # # print(viz_data[0:5])
    # with open(viz_data_file, 'w') as f:
    #     json.dump(viz_data, f)
    # f.close()

    clusters_viz.to_json(viz_data_file, orient="records")


if __name__ == '__main__':
    merged_meta_file = "index/dataset1.ivf1024.meta"
    results_pattern = "index/dataset1.ivf1024.mining_*"
    viz_data_file = "index/dataset1.ivf1024.mining.clusters.json"
    generate_clusters()