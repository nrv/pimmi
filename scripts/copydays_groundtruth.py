import os
import glob
import casanova

path_to_images = "/var/www/html/pimmi-copydays/copydays/"
path_to_clusters_file = "/var/www/html/pimmi-copydays/index/copydays.IVF1024,Flat.mining.clusters.csv"
prefix = len(path_to_images.strip("/"))

def create_row(cluster_dict, relative_path):
    return [relative_path, cluster_dict.get(relative_path, ""), relative_path[-10:-6]]

images_to_pimmi_clusters = {}
with open(path_to_clusters_file, "r") as f:
    reader = casanova.reader(f)
    path_pos = reader.headers["path"]
    cluster_pos = reader.headers["cluster_id"]
    for row in reader:
        images_to_pimmi_clusters[row[path_pos]] = row[cluster_pos]

transformed_images = []
with open(path_to_clusters_file.replace("clusters", "groundtruth"), "w") as f:
    writer = casanova.writer(f, fieldnames=["path", "predicted", "truth", "image_status"])
    for image in glob.glob(os.path.join(path_to_images, "copydays_original", "*.jpg")):
        relative_path = image[prefix+2:]
        writer.writerow(
            create_row(images_to_pimmi_clusters, relative_path) + ["original"]
        )
    for image in glob.glob(os.path.join(path_to_images, "**", "*.jpg"), recursive=True):
        relative_path = image[prefix+2:]
        if "original" not in relative_path:
            writer.writerow(
                create_row(images_to_pimmi_clusters, relative_path) + ["copy"]
            )
