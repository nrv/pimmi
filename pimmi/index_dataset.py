import logging
import pimmi
import pimmi_parameters as prm
import pimmi_toolbox as tbx
import argparse

logger = logging.getLogger("index")

index_type = "IVF1024,Flat"


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--action", required=True, choices=["create_empty", "fill", "info", "correct"])
parser.add_argument('--index', required=False, help="faiss factory string, default : " + index_type)
parser.add_argument('--thread', required=False, type=int, help="nb threads, default : " + str(prm.nb_threads))
parser.add_argument('--nb_img', required=False, type=int, help="nb images to train index (default : " +
                                                               str(prm.nb_images_to_train_index) +
                                                               ") or to add to index")
parser.add_argument('--load_faiss', required=False, help="faiss index file to load")
parser.add_argument('--save_faiss', required=False, help="faiss index file to save")
image_source = parser.add_mutually_exclusive_group()
image_source.add_argument('--images_dir', required=False, help="index all images from this directory")
image_source.add_argument('--images_meta', required=False, help="index all images from this file")
parser.add_argument('--images_meta_root', required=False, help="root path for meta relative paths")
args = parser.parse_args()

if __name__ == '__main__':
    if args.thread:
        prm.nb_threads = args.thread
    if args.nb_img:
        prm.nb_images_to_train_index = args.nb_img
    if args.index:
        index_type = args.index

    logger.info("action executed : " + args.action)

    if "info" == args.action:
        if args.load_faiss:
            previous_faiss_index = args.load_faiss + ".faiss"
            previous_faiss_meta = args.load_faiss + ".meta"
            previous_index = pimmi.load_index(previous_faiss_index, previous_faiss_meta)
        exit(0)

    if "correct" == args.action:
        if args.load_faiss:
            previous_faiss_index = args.load_faiss + ".faiss"
            previous_faiss_meta = args.load_faiss + ".meta"
            previous_index = pimmi.load_index(previous_faiss_index, previous_faiss_meta, correct=index_type)
            pimmi.save_index(previous_index, previous_faiss_index, previous_faiss_meta)
        exit(0)

    if not args.save_faiss:
        logger.error("save_faiss should be provided")
        exit(1)

    logger.info("using " + str(prm.nb_threads) + " threads")

    images = None
    images_root = None
    if args.images_dir:
        logger.info("listing images recursively from : " + args.images_dir)
        images = tbx.get_all_images([args.images_dir])
        images_root = args.images_dir
    if args.images_meta:
        if not args.images_meta_root:
            logger.error("images_meta_root should be provided")
            exit(1)
        logger.info("getting images from : " + args.images_meta)
        images = tbx.get_all_retrieved_images(args.images_meta, args.images_meta_root)
        images_root = args.images_meta_root
    if images is None:
        logger.error("you should provide a way to find images")
        exit(1)

    if args.nb_img:
        images = images[0:args.nb_img]

    if "create_empty" == args.action:
        if args.load_faiss:
            logger.error("unable to load an index before creating an empty one")
            exit(1)
        logger.info("index type : " + index_type)
        logger.info("using " + str(prm.nb_images_to_train_index) + " images to train index")
        empty_index = pimmi.create_index_mt(index_type, images, images_root, only_empty_index=True)
        empty_faiss_index = args.save_faiss + ".faiss"
        empty_faiss_meta = args.save_faiss + ".meta"
        pimmi.save_index(empty_index, empty_faiss_index, empty_faiss_meta)

    if "fill" == args.action:
        if args.load_faiss:
            previous_faiss_index = args.load_faiss + ".faiss"
            previous_faiss_meta = args.load_faiss + ".meta"
            previous_index = pimmi.load_index(previous_faiss_index, previous_faiss_meta)
            logger.info("using " + str(prm.nb_images_to_train_index) + " images to fill index")
            filled_index = pimmi.fill_index_mt(previous_index, images, images_root)
        else:
            logger.info("using " + str(prm.nb_images_to_train_index) + " images to train and fill index")
            filled_index = pimmi.create_index_mt(index_type, images, images_root)
        filled_faiss_index = args.save_faiss + ".faiss"
        filled_faiss_meta = args.save_faiss + ".meta"
        pimmi.save_index(filled_index, filled_faiss_index, filled_faiss_meta)
