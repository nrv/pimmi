import logging
import argparse
import casanova
import json
from json import JSONEncoder

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("viz")


def obj_dict(obj):
    return obj.__dict__


class Cluster:
    def __init__(self):
        self.cluster: int = -1
        self.match_quality: float = -1
        self.sample_path: str = ""
        self.images = []

    def __repr__(self):
        return 'cluster ' + str(self.cluster) + ' (' + str(len(self.images)) + ' images)'


def from_clusters_to_viz(clusters_file, viz_file):
    logger.info("Loading %s", clusters_file)
    clusters = {}
    with open(clusters_file) as infile:
        reader = casanova.reader(infile)
        path_pos = reader.headers.path
        cluster_id_pos = reader.headers.cluster_id
        quality_pos = reader.headers.quality
        for row in reader:
            cluster_id = row[cluster_id_pos]
            if cluster_id in clusters:
                cluster = clusters[cluster_id]
            else:
                cluster = Cluster()
                cluster.cluster = cluster_id
                cluster.match_quality = row[quality_pos]
                cluster.sample_path = row[path_pos]
                clusters[cluster_id] = cluster
            cluster.images.append(row[path_pos])
        reader.close()
    logger.info("Writing %s", viz_file)
    with open(viz_file, 'w') as outfile:
        outfile.write(json.dumps(list(clusters.values()), default=obj_dict, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--clusters', required=True, help="input clusters CSV file")
    parser.add_argument('--viz', required=True, help="output viz JSON file")
    args = parser.parse_args()

    from_clusters_to_viz(args.clusters, args.viz)
