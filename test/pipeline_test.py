# =============================================================================
# Pimmi Functional Tests
# =============================================================================
import csv
from glob import glob
from collections import defaultdict
from os import remove
from os.path import join, dirname

from test.utils import prm

SMALL_DATASET_QUERY_RESULTS = join(dirname(__file__), "ressources", "query_results.csv")


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


class TestPipeline(object):
    def test_fill_query(self):
        tmp_folder_path = join(dirname(__file__), "ressources", "tmp")
        results = load_query_results_from_file(SMALL_DATASET_QUERY_RESULTS)
        tested_results = load_query_results_from_file(join(tmp_folder_path, "small.IDMap,Flat.mining_000000.csv"))

        for query in results:
            assert query in tested_results, 'The line corresponding to query %s is missing' % (query)
            for result in results[query]:
                assert result in tested_results[query], 'The line corresponding to query, resultp pair ' \
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
        for file in glob(join(tmp_folder_path, "*")):
            remove(file)

