# =============================================================================
# Pimmi Functional Tests
# =============================================================================
import csv
from glob import glob
from collections import defaultdict
from os import remove
from os.path import join, dirname

from pimmi.cli.config import parameters as prm

SMALL_DATASET_QUERY_RESULTS = join(dirname(__file__), "ressources", "query_results.csv")

config_path = join(dirname(dirname(__file__)), "pimmi", "cli", "config.yml")
config_dict = prm.load_config_file(config_path)
prm.set_config_as_attributes(config_dict)


def load_query_results_from_file(file):
    fieldnames = ["keep","query_nb_points","result_image_id","nb_match_total","keep_smn","nb_match_ransac","keep_rns",
                  "ransac_ratio","result_path","result_width","result_height","result_nb_points","query_path",
                  "query_width","query_height","query_image_id","pack_id"]
    selected_names = ["keep","query_nb_points","result_image_id",#"nb_match_total",
                      #"keep_smn", "nb_match_ransac", "ransac_ratio",
                      "keep_rns","result_path","result_width","result_height","result_nb_points",
                      "query_path", "query_width","query_height","query_image_id","pack_id"]

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
        # fill(SMALL_DATASET_DIR, "small", tmp_folder_path, config_path=None, index_type = 'IVF1024,Flat',
        #      erase=False, force=False, nb_threads=1)
        # query("small", SMALL_DATASET_DIR, tmp_folder_path, config_path=None, nb_per_split=10000, simple=True, nb_threads=1)
        tested_results = load_query_results_from_file(join(tmp_folder_path, "small.IVF1024,Flat.mining_000000.csv"))

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

