#!/usr/bin/env python
# =============================================================================
# Pimmi CLI Endpoint
# =============================================================================
#
# CLI enpoint of the Pimmi library.

import os
import csv
import sys
import yaml
import logging
import argparse

from pimmi import load_index, save_index, fill_index_mt, create_index_mt, get_index_images, query_index_mt
import pimmi.toolbox as tbx
from pimmi.cli.config import parameters as prm

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(prog="pimmi", description='PIMMI: a command line tool for image mining.')
prm.load_config_file(parser, "pimmi/cli/config.yml")


def load_cli_parameters():
    subparsers = parser.add_subparsers(title="commands")
    parser_fill = subparsers.add_parser('fill', help="Create index and fill with vectors of images. Receive IMAGE-DIR, "
                                                     "a directory containing images. Index these images and save. "
                                                     "INDEX-NAME will be used as index id by other pimmi commands.")

    parser_query = subparsers.add_parser('query', help="Query some index.")

    parser_fill.add_argument('image_dir', type=str, metavar='image-dir')
    parser_fill.add_argument('index_name', type=str, metavar='index-name')
    parser_fill.add_argument("--index-path", type=str, help="Directory where the index should be stored/loaded from. "
                                                            "Defaults to './index'", default="./index")
    parser_fill.add_argument("--index-type", type=str, choices=["IVF1024,Flat"], default="IVF1024,Flat")
    parser_fill.add_argument("--load-faiss", action="store_true", default=False)
    parser_fill.set_defaults(func=fill)

    cli_parameters = vars(parser.parse_args(namespace=prm))
    return cli_parameters


def fill(image_dir, index_name, index_path, load_faiss, **kwargs):

    if not os.path.isdir(image_dir):
        logger.error("The provided image-dir is not a directory.")
        sys.exit(1)

    if not index_name:
        index_name = os.path.basename(os.path.normpath(image_dir))

    logger.info("listing images recursively from : " + image_dir)
    images = tbx.get_all_images([image_dir])

    if load_faiss:
        previous_faiss_index = os.path.join(index_path, ".".join([index_name, prm.index_type, "faiss"]))
        previous_faiss_meta = os.path.join(index_path, ".".join([index_name, prm.index_type, "meta"]))
        previous_index = load_index(previous_faiss_index, previous_faiss_meta)
        logger.info("using " + str(prm.nb_images_to_train_index) + " images to fill index")
        filled_index = fill_index_mt(previous_index, images, image_dir)
    else:
        logger.info("using " + str(prm.nb_images_to_train_index) + " images to train and fill index")
        filled_index = create_index_mt(prm.index_type, images, image_dir)

    if not os.path.isdir(index_path):
        print("Are you sure you want to save index data in {}? y/n".format(os.path.abspath(index_path)))
        if input() == 'y':
            os.mkdir(index_path)
        else:
            print('Please enter a valid directory')
            index_path = input()
            if not os.path.isdir(index_path):
                logger.error("{} does not exist.".format(os.path.abspath(index_path)))
                sys.exit(1)

    filled_faiss_index = os.path.join(index_path, ".".join([index_name, prm.index_type, "faiss"]))
    filled_faiss_meta = os.path.join(index_path, ".".join([index_name, prm.index_type, "meta"]))
    save_index(filled_index, filled_faiss_index, filled_faiss_meta)


def query(index_name, image_dir, index_path, index_type, nb_img, nb_per_split, simple, **kwargs):
    faiss_index = os.path.join(index_path, ".".join([index_name, index_type, "faiss"]))
    faiss_meta = os.path.join(index_path, ".".join([index_name, index_type, "meta"]))
    index = load_index(faiss_index, faiss_meta)

    images = get_index_images(index, image_dir)

    if nb_img:
        images = images.head(nb_img)

    logger.info("total number of queries " + str(len(images)))
    images = images.sort_values(by=prm.dff_image_path)
    queries = tbx.split_pack(images, nb_per_split)
    for pack in queries:
        pack_result_file = faiss_index.replace("faiss", "mining") + "_" + str(pack[prm.dff_pack_id]).zfill(6) + ".csv"
        logger.info("query " + str(len(pack[prm.dff_pack_files])) + " files from pack " +
                    str(pack[prm.dff_pack_id]) + " -> " + pack_result_file)
        query_result = query_index_mt(index, pack[prm.dff_pack_files], image_dir, pack=pack[prm.dff_pack_id])
        query_result = query_result.sort_values(by=[prm.dff_query_path, prm.dff_nb_match_ransac, prm.dff_ransac_ratio],
                                                ascending=False)
        query_result.to_csv(pack_result_file, index=False, quoting=csv.QUOTE_NONNUMERIC)


def main():
    cli_params = load_cli_parameters()
    if "func" in cli_params:
        command = cli_params.pop("func")
        command(**cli_params)
    else:
        parser.print_help()





