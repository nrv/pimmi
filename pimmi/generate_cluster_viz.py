import pimmi_parameters as prm
import logging
import pickle
import pandas as pd
import numpy as np
import igraph as ig
import json
import pimmi

# TODO parameters from command line
logger = logging.getLogger("gener")

merged_meta_file = "index/dataset1.ivf1024.meta"
all_results_file = "index/dataset1.ivf1024.mining.all"
viz_data_file = "index/dataset1.ivf1024.mining.clusters.json"


def from_cluster_2col_table_to_list(data, keycol, valcol):
    keys, values = data.sort_values(keycol).values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    return pd.DataFrame({keycol: ukeys, valcol: [list(a) for a in arrays]})


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

    logger.info("Loading results from " + all_results_file)
    with open(all_results_file, 'rb') as f:
        all_similarities = pickle.load(f)
    f.close()
    logger.info(" ~ similarity file : " + str(len(all_similarities)) + " edges")
    # print(all_similarities.head(5))

    with open(merged_meta_file, 'rb') as f:
        all_meta = pickle.load(f)
    f.close()
    all_meta = pimmi.meta_as_df(all_meta)
    logger.info("all_meta : %d", len(all_meta))

    # all_meta = all_meta.sort_values(by=[prm.dff_nb_points], ascending=True)
    # TODO check comment below
    # all_meta = all_meta[~pd.isna(all_meta[prm.dff_nb_points])]
    logger.info("all_meta filtered : %d", len(all_meta))
    # print(all_meta.head(5))

    all_similarities = all_similarities[["query_image_id", "result_image_id", "nb_match_ransac"]]

    all_results_with_meta = pd.merge(all_similarities, all_meta, how="left",
                                     left_on="query_image_id", right_on="image_id")
    all_results_with_meta = pd.merge(all_results_with_meta, all_meta, how="left", left_on="result_image_id",
                                     right_on="image_id", suffixes=("_query", "_result"))
    all_results_with_meta["query_ransac_ratio"] = all_results_with_meta["nb_match_ransac"] / \
                                                  all_results_with_meta["nb_points_query"]
    all_results_with_meta["result_ransac_ratio"] = all_results_with_meta["nb_match_ransac"] / \
                                                   all_results_with_meta["nb_points_result"]
    logger.info("all_results_with_meta : %d edges", len(all_results_with_meta))

    all_results_with_meta_graph = all_results_with_meta.query('nb_match_ransac >= 10')
    # all_results_with_meta_graph = all_results_with_meta.query('nb_match_ransac >= 10 and (query_ransac_ratio > 0.8 or result_ransac_ratio > 0.8)')
    # all_results_with_meta_graph = all_results_with_meta.query('nb_match_ransac >= 300 and (query_ransac_ratio > 0.5 or result_ransac_ratio > 0.5)')
    logger.info("all_results_with_meta_graph : %d edges", len(all_results_with_meta_graph))
    # print(all_results_with_meta_graph.head(100))

    g = ig.Graph.TupleList([(row["query_image_id"], row["result_image_id"], row["nb_match_ransac"])
                            for index, row in all_results_with_meta_graph.iterrows()], directed=True, weights=True)

    logger.info("Number of vertices in the graph: %d", g.vcount())
    logger.info("Number of edges in the graph: %d", g.ecount())
    logger.info("Is the graph directed: %d", g.is_directed())
    logger.info("Maximum degree in the graph: %d", g.maxdegree())


    comp = g.components(mode="weak")
    logger.info("Connected components in the graph: %d", len(comp))
    clusters = pd.DataFrame(list(zip(g.vs["name"], comp.membership)), columns=['image_id', 'cluster'])
    logger.info("Images in clusters : %d", len(clusters))

    clusters_with_meta = pd.merge(clusters, all_meta, how="left", left_on="image_id", right_on="image_id")
    # print(clusters_with_meta.head(10))

    # summary_clusters = clusters_with_meta.groupby("cluster").agg({prm.mex_nbSeen: ['sum', 'count'],
    #                                                               prm.mex_relative_path: ['first']})
    # summary_clusters.columns = ['nb_seen', 'nb_images', 'sample_path']
    # summary_clusters = summary_clusters.reset_index()
    # logger.info("%d distinct clusters", len(summary_clusters))
    # summary_clusters = summary_clusters.sort_values(by=['nb_seen', 'nb_images'], ascending=False)

    summary_clusters = clusters_with_meta.groupby("cluster").agg({prm.dff_image_path: ['first', 'count']})
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
        logger.info("cluster %d has %d vertices and %d edges, quality index : %f", cluster_id, sg.vcount(), sg.ecount(), quality)

    print(summary_clusters.head(10))

    clusters_viz = from_cluster_2col_table_to_list(clusters_with_meta[["cluster", prm.dff_image_path]], "cluster", "images")
    clusters_viz = pd.merge(clusters_viz, summary_clusters, how="left", left_on="cluster", right_on="cluster")
    clusters_viz = clusters_viz.sort_values(by=['nb_images'], ascending=False)
    print(clusters_viz.head(10))

    # viz_data = list(clusters_viz["relativePath"])
    # # print(viz_data[0:5])
    # with open(viz_data_file, 'w') as f:
    #     json.dump(viz_data, f)
    # f.close()

    clusters_viz.to_json(viz_data_file, orient="records")
