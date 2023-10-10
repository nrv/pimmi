# =============================================================================
# Pimmi Functional Tests
# =============================================================================
import csv
import logging
from os import remove
import glob
from shutil import rmtree
from collections import defaultdict
from os.path import join, dirname, isfile, isdir

RESSOURCES_PATH = join(dirname(__file__), "ressources")
SMALL_DATASET_QUERY_RESULTS = join(RESSOURCES_PATH, "query_results.csv")
SMALL_DATASET_CLUSTERING_RESULTS = join(
    RESSOURCES_PATH, "clusters_results.csv")
TMP_FOLDER_PATH = join(RESSOURCES_PATH, "tmp")

logger = logging.getLogger("pimmi")


def load_query_results_from_file(files):
    fieldnames = ["keep", "query_nb_points", "result_image_id", "nb_match_total", "keep_smn", "nb_match_ransac",
                  "keep_rns", "ransac_ratio", "result_path", "result_width", "result_height", "result_nb_points",
                  "query_path", "query_width", "query_height", "query_image_id", "pack_id"]
    selected_names = ["query_nb_points", "nb_match_total",
                      # new file version - "keep", "keep_smn"
                      # ransac stuff has a random part - "nb_match_ransac", "ransac_ratio", "keep_rns",
                      # image_id may change - "result_image_id", "query_image_id"
                      "result_path", "result_width", "result_height", "result_nb_points",
                      "query_path", "query_width", "query_height", "pack_id"]

    query_results = defaultdict(lambda: defaultdict(dict))
    for file in files:

        with open(file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for column in selected_names:
                    query_results[row["query_path"]
                                  ][row["result_path"]][column] = row[column]

    return query_results


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
        results = load_query_results_from_file(
            [SMALL_DATASET_QUERY_RESULTS])
        tested_results = load_query_results_from_file(
            glob.glob(join(TMP_FOLDER_PATH, "small_queries*")))

        for query in results:

            assert query in tested_results, 'The line corresponding to query %s is missing' % (
                query)
            for result in results[query]:
                assert result in tested_results[query], 'The line corresponding to query, result pair ' \
                    '(%s, %s) is missing' % (
                    query, result)
                for column in results[query][result]:
                    assert column in tested_results[query][result], 'missing column "%s" in tested file for query, ' \
                        'result pair (%s, %s)' % (
                        column, query, result)
                    assert results[query][result][column] == tested_results[query][result][column], 'Different values' \
                        ' for column "%s"' \
                        ' of query, result'\
                        ' pair (%s, %s)'\
                        % (column, query, result)

    def test_cluster(self):
        results = load_clusters_results_from_file(
            SMALL_DATASET_CLUSTERING_RESULTS)
        tested_results = load_clusters_results_from_file(
            join(TMP_FOLDER_PATH, "small_clusters.csv"))

        for image, row in results.items():
            assert image in tested_results, 'Image %s is missing' % (image)
            assert tested_results[image] == row

        for file in glob.glob(join(TMP_FOLDER_PATH, "*")):
            if isfile(file):
                remove(file)
            elif isdir(file):
                rmtree(file)
