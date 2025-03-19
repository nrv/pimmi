# =============================================================================
# Pimmi Functional Tests
# =============================================================================
import csv
import math
import sys
from os import remove, listdir
import glob
from shutil import rmtree
from collections import defaultdict
from os.path import join, dirname, isfile, isdir

RESSOURCES_PATH = join(dirname(__file__), "ressources")
# do this to use files that have windows path in it
OS_SUFFIX = "_win32" if sys.platform == "win32" else ""
SMALL_DATASET_QUERY_RESULTS = join(RESSOURCES_PATH, f"query_results{OS_SUFFIX}.csv")
SMALL_DATASET_CLUSTERING_RESULTS = join(
    RESSOURCES_PATH, f"clusters_results{OS_SUFFIX}.csv")
TMP_FOLDER_PATH = join(RESSOURCES_PATH, "tmp")
TMP_FOLDER_EMPTY = len(listdir(TMP_FOLDER_PATH)) == 1
NB_LINES_PER_FILE = 30


def load_query_results_from_files(files):
    nb_lines = 0
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
                nb_lines += 1
                for column in selected_names:
                    query_results[row["query_path"]
                                  ][row["result_path"]][column] = row[column]

    return query_results, nb_lines


def load_clusters_results_from_file(file):
    clusters_results = {}

    with open(file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            clusters_results[row[0]] = row[2:]

    return clusters_results


def assert_query(query, results, tested_file):
    assert query in tested_file, 'The line corresponding to query %s is missing' % (
    query)

    for result in results[query]:
        assert result in tested_file[query], 'The line corresponding to query, result pair ' \
            '(%s, %s) is missing' % (
            query, result)
        for column in results[query][result]:
            assert column in tested_file[query][result], 'missing column "%s" in tested file for query, ' \
                'result pair (%s, %s)' % (
                column, query, result)
            assert results[query][result][column] == tested_file[query][result][column], 'Different values' \
                ' for column "%s"' \
                ' of query, result'\
                ' pair (%s, %s)'\
                % (column, query, result)


class TestPipeline(object):
    def test_query(self):
        if TMP_FOLDER_EMPTY:
            return
        results, expected_nb_lines = load_query_results_from_files(
            [SMALL_DATASET_QUERY_RESULTS])
        tested_results_unique_file, nb_lines_unique_file = load_query_results_from_files(
            [join(TMP_FOLDER_PATH, "small_queries.csv")]
            )

        expected_nb_files = math.ceil(expected_nb_lines / NB_LINES_PER_FILE)
        tested_results_multiple_files, nb_lines_multiple_files = load_query_results_from_files(
            [join(TMP_FOLDER_PATH, "small_queries{}.csv").format(i) for i in range(0, expected_nb_files)]
            )

        assert nb_lines_unique_file == nb_lines_multiple_files

        for query in results:
            assert_query(query, results, tested_results_unique_file)
            assert_query(query, results, tested_results_multiple_files)

    def test_cluster(self):
        if TMP_FOLDER_EMPTY:
            return
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
