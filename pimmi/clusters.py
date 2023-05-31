import sys
import glob
import json
import pickle
import logging
import casanova
import networkit as nk

logger = logging.getLogger("pimmi")


def obj_dict(obj):
    return obj.__dict__


class ClusterViz:
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
                cluster = ClusterViz()
                cluster.cluster = cluster_id
                cluster.match_quality = row[quality_pos]
                cluster.sample_path = row[path_pos]
                clusters[cluster_id] = cluster
            cluster.images.append(row[path_pos])
        reader.close()
    logger.info("Writing %s", viz_file)
    with open(viz_file, 'w') as outfile:
        outfile.write(json.dumps(list(clusters.values()), default=obj_dict, indent=4))


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
    cc = nk.components.WeaklyConnectedComponents(g) if g.isDirected() \
        else nk.components.ConnectedComponents(g)
    cc.run()
    logger.info("Connected components in the graph: %d", cc.numberOfComponents())
    components = cc.getComponents()

    community_counter = 0
    for component_id, comp in enumerate(components):
        sg = nk.graphtools.subgraphFromNodes(g, comp)
        if algo == "components":
            yield component_id, sg

        elif algo == "louvain":
            algo = nk.community.PLM(sg, refine=True, turbo=False)
            algo.run()
            partition = algo.getPartition()
            for community_id in partition.getSubsetIds():
                ssg = nk.graphtools.subgraphFromNodes(g, partition.getMembers(community_id))
                community_counter += 1
                yield community_counter, ssg


    if algo == "louvain":
        logger.info("Louvain communities in the graph: %d", community_counter)


def generate_clusters(results_pattern, merged_meta_file, clusters_file, nb_match_ransac, algo):
    logger.info("Loading query results")

    if algo == "louvain":
        g = nk.graph.Graph(weighted=True)
        for u, v, weight in generate_graph_from_files(results_pattern, nb_match_ransac):
            if g.hasEdge(u, v):
                g.increaseWeight(u, v, weight)
            else:
                g.addEdge(u, v, w=weight, addMissing=True)
    elif algo == "components":
        g = nk.graph.Graph(directed=True, weighted=True)
        for u, v, weight in generate_graph_from_files(results_pattern, nb_match_ransac):
            g.addEdge(u, v, w=weight, addMissing=True)

    else:
        raise ValueError("'algo' must be set to 'louvain' or 'components'")


    logger.info("Graph size: {} nodes, {} edges".format(*nk.graphtools.size(g)))

    with open(merged_meta_file, 'rb') as f:
        meta_json = pickle.load(f)

    f = open(clusters_file, 'w') if clusters_file else sys.stdout

    writer = casanova.writer(f, ["path", "image_id", "nb_points", "degree", "cluster_id", "quality"])
    for community_id, sg in yield_communities(g, algo):
        nb_matches = sg.totalEdgeWeight()
        nb_points = []
        paths = []
        degrees = []

        for node_id in sg.iterNodes():
            meta_image = meta_json[node_id]
            nb_points.append(meta_image["nb_points"])
            paths.append(meta_image["path"])
            degrees.append(sg.degreeIn(node_id) + sg.degreeOut(node_id))

        sum_weight = range(len(nb_points), 0, -1)
        max_theoretical_matches = 2 * sum([i * j for i, j in zip(sorted(nb_points), sum_weight)])
        quality = nb_matches / max_theoretical_matches

        for node_id, nb, path, degree in zip(sg.iterNodes(), nb_points, paths, degrees):
            writer.writerow([path, node_id, nb, degree, community_id, quality])

    f.close()


if __name__ == '__main__':
    merged_meta_file = "index/dataset1.IVF1024,Flat.meta"
    results_pattern = "index/dataset1.IVF1024,Flat.mining_*"
    clusters_file = "index/dataset1.IVF1024,Flat.mining.clusters.csv"

    generate_clusters(
        results_pattern,
        merged_meta_file,
        clusters_file,
        nb_match_ransac=10,
        algo="louvain"
    )
