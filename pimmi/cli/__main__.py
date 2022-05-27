#!/usr/bin/env python
# =============================================================================
# Pimmi CLI Endpoint
# =============================================================================
#
# CLI enpoint of the Pimmi library.
#
import os
import csv
import sys
import click
import logging

from pimmi import load_index, save_index, fill_index_mt, create_index_mt, get_index_images, query_index_mt
import pimmi.pimmi_parameters as prm
import pimmi.toolbox as tbx

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"]
}

logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@main.command(help="Create index and fill with vectors of images. Receive IMAGE-DIR, a directory containing images. "
                   "Index these images and save. INDEX-NAME will be used as index id by other pimmi commands.")
@click.argument("image-dir", type=click.Path(exists=True))
@click.argument("index-name", type=str)
@click.option("--index-path", type=str, help="Directory where the index should be stored/loaded from. "
                                             "Defaults to './index'", default="./index")
@click.option("--index-type", type=str, default="IVF1024,Flat")
@click.option("--load-faiss", is_flag=True, type=bool)
def fill(image_dir, index_name, index_path, index_type, load_faiss):

    if not os.path.isdir(image_dir):
        logger.error("The provided image-dir is not a directory.")
        sys.exit(1)

    if not index_name:
        index_name = os.path.basename(os.path.normpath(image_dir))

    logger.info("listing images recursively from : " + image_dir)
    images = tbx.get_all_images([image_dir])

    if load_faiss:
        previous_faiss_index = os.path.join(index_path, ".".join([index_name, index_type, "faiss"]))
        previous_faiss_meta = os.path.join(index_path, ".".join([index_name, index_type, "meta"]))
        previous_index = load_index(previous_faiss_index, previous_faiss_meta)
        logger.info("using " + str(prm.nb_images_to_train_index) + " images to fill index")
        filled_index = fill_index_mt(previous_index, images, image_dir)
    else:
        logger.info("using " + str(prm.nb_images_to_train_index) + " images to train and fill index")
        filled_index = create_index_mt(index_type, images, image_dir)

    if not os.path.isdir(index_path):
        if click.confirm("Are you sure you want to save index data in {}?".format(os.path.abspath(index_path))):
            os.mkdir(index_path)
        else:
            index_path = click.prompt('Please enter a valid directory', type=click.Path(exists=True))
            if not os.path.isdir(index_path):
                logger.error("{} does not exist.".format(os.path.abspath(index_path)))
                sys.exit(1)

    filled_faiss_index = os.path.join(index_path, ".".join([index_name, index_type, "faiss"]))
    filled_faiss_meta = os.path.join(index_path, ".".join([index_name, index_type, "meta"]))
    save_index(filled_index, filled_faiss_index, filled_faiss_meta)


@main.command(help="Query some index.")
@click.argument("index-name", type=str)
@click.argument("image-dir", type=click.Path(exists=True))
@click.option("--index-path", type=str, help="Directory where the index should be loaded from. "
                                             "Defaults to './index'", default="./index")
@click.option("--index-type", type=str, default="IVF1024,Flat")
@click.option("--nb-img", type=int, help="Nb images to query the index. Defaults to the total nb of images.")
@click.option('--nb_per_split', type=int, help="nb images to query per pack", default=10000)
@click.option('--simple', help="use only very simple query params", is_flag=True, type=bool)
def query(index_name, image_dir, index_path, index_type, nb_img, nb_per_split, simple):
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
