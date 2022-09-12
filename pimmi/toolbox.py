import glob
import json
import math
import cv2 as cv
import pandas as pd
from os.path import join
import pimmi.pimmi_parameters as constants
import logging

logger = logging.getLogger("tools")

ALLOWLIST_FORMATS = (".jpeg", ".jpg", ".png")


def get_all_images(image_dir):
    files = []
    for extension in ALLOWLIST_FORMATS:
        files.extend(glob.glob(join(image_dir, "**", "*{}".format(extension)), recursive=True))
    return files


def parse_mex(file):
    with open(file, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
    f.close()


def get_all_metadata(file, only_retrieved=False):
    logger.info("Reading " + file)
    all_data = list(parse_mex(file))
    all_data = pd.json_normalize(all_data)
    logger.info(" ~ found " + str(len(all_data)) + " images metadata")
    if only_retrieved:
        all_data = all_data[~pd.isna(all_data[constants.mex_ext_retrieved])]
    return all_data


# TODO: change to (id, path) df
def get_all_retrieved_images(file, path_prefix):
    all_data = get_all_metadata(file, only_retrieved=True)
    all_data = all_data[[prm.mex_relative_path]]
    all_data["temp"] = path_prefix
    all_data[constants.dff_image_path] = all_data["temp"] + all_data[constants.mex_relative_path]
    logger.info(" ~ keeping " + str(len(all_data)) + " retrieved images")
    return all_data[constants.dff_image_path].tolist()


def split_pack(all_files, size=-1):
    splits = []

    if size <= 0:
        splits.append({constants.dff_pack_id: 0, constants.dff_pack_files: all_files})
        return splits

    nb_files = len(all_files)
    nb_packs = math.ceil(nb_files / size)

    for p in range(nb_packs):
        pack_files = all_files[p * size:(p + 1) * size]
        splits.append({constants.dff_pack_id: p, constants.dff_pack_files: pack_files})

    return splits


class Sift(object):
    def __init__(
            self,
            sift_nfeatures,
            sift_nOctaveLayers,
            sift_contrastThreshold,
            sift_edgeThreshold,
            sift_sigma,
            nb_threads
    ):

        self.config = cv.SIFT_create(nfeatures=sift_nfeatures, nOctaveLayers=sift_nOctaveLayers,
                              contrastThreshold=sift_contrastThreshold, edgeThreshold=sift_edgeThreshold,
                              sigma=sift_sigma)
        logger.info("Using opencv : " + cv.__version__)
        cv.setNumThreads(nb_threads)