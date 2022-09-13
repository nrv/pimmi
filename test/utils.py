import csv
import cv2 as cv
import numpy as np
from os.path import join, dirname

from pimmi.pimmi import resize_if_needed, point_to_full_id
from pimmi.cli.config import parameters as prm


SMALL_DATASET_DIR = join(dirname(dirname(__file__)), 'demo_dataset', 'small_dataset')
IMAGE_PATH = join(SMALL_DATASET_DIR, "000010.jpg")
EXAMPLE_FILE = join(dirname(__file__), "ressources", "image_000010_sifts.csv")
IMAGE_ID = 0

config_path = join(dirname(dirname(__file__)), "pimmi", "cli", "config.yml")
config_dict = prm.load_config_file(config_path)
prm.set_config_as_attributes(config_dict)

kp_fieldnames = ["x", "y", "angle", "class_id", "octave", "response", "size"]
kp_types = [float, float, float, int, int, float, float]

sift = cv.SIFT_create(nfeatures=1000, nOctaveLayers=1, contrastThreshold=0.1, edgeThreshold=10, sigma=1.6)


def recreate_sifts():
    img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)
    img, resized = resize_if_needed(img)
    kp, desc = sift.detectAndCompute(img, None)

    ids = np.empty(len(kp), dtype="int64")

    for sift_point in range(len(kp)):
        ids[sift_point] = point_to_full_id(IMAGE_ID, kp[sift_point].pt, img.shape[1], img.shape[0])
    return desc, ids, kp


def write_sifts(desc, ids, kp):
    desc_ids = list(range(desc.shape[1]))
    fieldnames = ["id"] + kp_fieldnames + desc_ids

    with open(EXAMPLE_FILE, "w") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for id, kp_item, desc_item in zip(ids, kp, desc):
            writer.writerow(
                [id] +
                [kp_item.pt[0], kp_item.pt[1]] +
                [getattr(kp_item, attribute) for attribute in kp_fieldnames[2:]] +
                list(desc_item)
            )


def load_sifts(file):
    lines = -1

    with open(file, "r") as f:
        for line in f:
            lines += 1

    with open(file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        ids = np.empty(lines, dtype="int64")
        kp = []
        nb_named_attributes = len(["id"] + kp_fieldnames)
        nb_desc = len(header) - nb_named_attributes
        desc = np.empty((lines, nb_desc), dtype=np.float32)

        for i, row in enumerate(reader):
            ids[i] = row[0]
            desc[i] = row[nb_named_attributes:]
            kp_dict = {
                attribute: set_type(row[j+1]) for j, (attribute, set_type) in enumerate(zip(kp_fieldnames, kp_types))}
            kp.append(cv.KeyPoint(**kp_dict))

        return ids, desc, tuple(kp)


if __name__ == "__main__":
    write_sifts(*recreate_sifts())