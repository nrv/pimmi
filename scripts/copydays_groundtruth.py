import os
import sys
import glob
import casanova


def generate_copydays_eval(path_to_images, path_to_clusters_file):
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

    with open(path_to_clusters_file.replace("clusters", "groundtruth"), "w", newline="") as f:
        writer = casanova.writer(f, fieldnames=["path", "predicted", "truth", "image_status"])
        for image in glob.glob(os.path.join(path_to_images, "*original", "*.jpg")):
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


if __name__ == '__main__':
    path_to_images = sys.argv[1]
    path_to_clusters_file = sys.argv[2]
    generate_copydays_eval(path_to_images, path_to_clusters_file)