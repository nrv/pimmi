import glob
import json
import math
import pandas as pd
import pimmi_parameters as prm
import logging

logger = logging.getLogger("tools")


def get_all_images(image_dirs):
    files = []
    for image_dir in image_dirs:
        files.extend(glob.glob(image_dir + "/**/*.jpg", recursive=True))
        files.extend(glob.glob(image_dir + "/**/*.png", recursive=True))
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
        all_data = all_data[~pd.isna(all_data[prm.mex_ext_retrieved])]
    return all_data


# TODO: change to (id, path) df
def get_all_retrieved_images(file, path_prefix):
    all_data = get_all_metadata(file, only_retrieved=True)
    all_data = all_data[[prm.mex_relative_path]]
    all_data["temp"] = path_prefix
    all_data[prm.dff_image_path] = all_data["temp"] + all_data[prm.mex_relative_path]
    logger.info(" ~ keeping " + str(len(all_data)) + " retrieved images")
    return all_data[prm.dff_image_path].tolist()


def split_pack(all_files, size=-1):
    splits = []

    if size <= 0:
        splits.append({prm.dff_pack_id: 0, prm.dff_pack_files: all_files})
        return splits

    nb_files = len(all_files)
    nb_packs = math.ceil(nb_files / size)

    for p in range(nb_packs):
        pack_files = all_files[p * size:(p + 1) * size]
        splits.append({prm.dff_pack_id: p, prm.dff_pack_files: pack_files})

    return splits
