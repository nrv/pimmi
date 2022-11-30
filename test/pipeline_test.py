# =============================================================================
# Pimmi Functional Tests
# =============================================================================
import csv
import json
from glob import glob
from collections import defaultdict
from os import remove
from os.path import join, dirname

RESSOURCES_PATH = join(dirname(__file__), "ressources")
SMALL_DATASET_QUERY_RESULTS = join(RESSOURCES_PATH, "query_results.csv")
SMALL_DATASET_CLUSTERING_RESULTS = join(RESSOURCES_PATH, "clusters_results.csv")
TMP_FOLDER_PATH = join(RESSOURCES_PATH, "tmp")


def load_query_results_from_file(file):
    fieldnames = ["keep","query_nb_points", "result_image_id", "nb_match_total", "keep_smn", "nb_match_ransac",
                  "keep_rns", "ransac_ratio", "result_path", "result_width", "result_height", "result_nb_points",
                  "query_path", "query_width", "query_height", "query_image_id", "pack_id"]
    selected_names = ["query_nb_points", "nb_match_total",
                      # new file version - "keep", "keep_smn"
                      # ransac stuff has a random part - "nb_match_ransac", "ransac_ratio", "keep_rns",
                      # image_id may change - "result_image_id", "query_image_id"
                      "result_path", "result_width", "result_height", "result_nb_points",
                      "query_path", "query_width", "query_height", "pack_id"]

    query_results = defaultdict(lambda: defaultdict(dict))

    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in selected_names:
                query_results[row["query_path"]][row["result_path"]][column] = row[column]
    return dict(query_results)


def load_clusters_results_from_file(file):
    clusters_results = {}

    with open(file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            clusters_results[row[0]] = row[2:]

    return clusters_results


class TestPipeline(object):
    def test_query(self):
        results = load_query_results_from_file(SMALL_DATASET_QUERY_RESULTS)
        tested_results = load_query_results_from_file(join(TMP_FOLDER_PATH, "small.IDMap,Flat.mining_000000.csv"))

        for query in results:
            assert query in tested_results, 'The line corresponding to query %s is missing' % (query)
            for result in results[query]:
                assert result in tested_results[query], 'The line corresponding to query, result pair ' \
                                                        '(%s, %s) is missing' % (query, result)
                for column in results[query][result]:
                    assert column in tested_results[query][result], 'missing column "%s" in tested file for query, ' \
                                                                    'result pair (%s, %s)' % (column, query, result)
                    assert results[query][result][column] == tested_results[query][result][column], 'Different values' \
                                                                                                    ' for column "%s"' \
                                                                                                    ' of query, result'\
                                                                                                    ' pair (%s, %s)'\
                                                                                                    % (column, query,
                                                                                                       result)

    def test_cluster(self):
        results = load_clusters_results_from_file(SMALL_DATASET_CLUSTERING_RESULTS)
        tested_results = load_clusters_results_from_file(join(TMP_FOLDER_PATH, "small.IDMap,Flat.mining.clusters.csv"))

        for image, row in results.items():
            assert image in tested_results, 'Image %s is missing' % (image)
            assert tested_results[image] == row

        for file in glob(join(TMP_FOLDER_PATH, "*")):
            remove(file)
