import pandas as pd
import csv
import glob
import pickle
import logging
import pimmi_parameters as prm
import pimmi

# TODO parameters from command line, fuse with generate_cluster_viz
logger = logging.getLogger("fuse ")

results_pattern = "index/dataset1.ivf1024.mining_0*.csv"
all_results_file = "index/dataset1.ivf1024.mining.all"


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)

all_similarities = []
for file in sorted(glob.glob(results_pattern)):
    logger.info(file)
    similarities = pd.read_csv(file, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(" ~ similarity file loaded  : " + str(len(similarities)) + " edges")
    similarities = similarities.query('query_image_id != result_image_id and nb_match_ransac >= 10')
    logger.info(" ~ similarity file filtered: " + str(len(similarities)) + " edges")
    all_similarities.append(similarities)
logger.info("Concatenating results from " + str(len(all_similarities)) + " files")
all_similarities = pd.concat(all_similarities)
all_similarities[prm.dff_result_path] = "/" + all_similarities[prm.dff_result_path]
all_similarities[prm.dff_query_path] = "/" + all_similarities[prm.dff_query_path]
# all_similarities = all_similarities.astype({prm.dff_query_image: int, prm.dff_result_image: int})
with open(all_results_file, 'wb') as f:
    pickle.dump(all_similarities, f, pickle.HIGHEST_PROTOCOL)
f.close()

# logger.info("Loading results from " + all_results_file)
# with open(all_results_file, 'rb') as f:
#     all_similarities = pickle.load(f)
# f.close()
# logger.info(" ~ similarity file : " + str(len(all_similarities)) + " edges")
# print(all_similarities.head(20))