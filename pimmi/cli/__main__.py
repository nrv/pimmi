#!/usr/bin/env python
# =============================================================================
# Pimmi CLI Endpoint
# =============================================================================
#
# CLI enpoint of the Pimmi library.
#
import os
import sys
import click
import logging

from pimmi import load_index, save_index, fill_index_mt, create_index_mt
import pimmi.pimmi_parameters as prm
import pimmi.toolbox as tbx

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"]
}

logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@main.command(help="Receive a directory containing images as parameter. "
                   "Index these images and save the resulting index.")
@click.argument("image-dir", type=click.Path(exists=True))
@click.option("--index-name", type=str, help="Name of the index. Defaults to the name of the image directory.")
@click.option("--index-path", type=str, help="Directory where the index should be stored/loaded from. "
                                             "Defaults to './index'", default="./index")
@click.option("--index-type", type=str, default="IVF1024,Flat")
@click.option("--load-faiss", type=bool, default=False, is_flag=True)
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
