import logging
import csv
import pimmi
import toolbox as tbx
import pimmi_parameters as prm
import argparse
import pandas as pd

nb_per_split = 10000

logger = logging.getLogger("query")

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--thread', required=False, type=int, help="nb threads, default : " + str(prm.nb_threads))
parser.add_argument('--nb_img', required=False, type=int, help="nb images to query the index")
parser.add_argument('--nb_per_split', required=False, type=int, help="nb images to query per pack")
parser.add_argument('--load_faiss', required=True, help="faiss index file to load")
parser.add_argument('--save_mining', required=True, help="mining file to save")
parser.add_argument('--simple', required=False, action='store_true', help="use only very simple query params")
parser.add_argument('--sub', required=False, action='store_true', help="sub-image matching")
image_source = parser.add_mutually_exclusive_group()
image_source.add_argument('--images_dir', required=False, help="query all images from this directory")
image_source.add_argument('--images_meta', required=False, help="query all images from this file")
image_source.add_argument('--images_mining', required=False, action='store_true', help="query all images from the index")
parser.add_argument('--images_root', required=False, help="root path for imagers relative paths")
args = parser.parse_args()


if __name__ == '__main__':
    if args.simple:
        # prm.nb_threads = 1
        prm.do_filter_on_sift_dist = False
        prm.adaptative_sift_nn = False

    if args.thread:
        prm.nb_threads = args.thread
    if args.nb_per_split:
        nb_per_split = args.nb_per_split

    prm.remove_query_from_results = False

    images = None
    images_root = None
    if not args.images_root:
        if args.images_meta or args.images_mining:
            logger.error("images_root should be provided")
            exit(1)

    if args.images_dir:
        logger.info("listing images recursively from : " + args.images_dir)
        images = tbx.get_all_images([args.images_dir])
        images_root = args.images_dir
    if args.images_meta:
        logger.info("getting images from : " + args.images_meta)
        # TODO: change to (id, path) df
        images = tbx.get_all_retrieved_images(args.images_meta, args.images_root)
        images_root = args.images_root
    if args.images_mining:
        logger.info("querying the indexed dataset on himself")
        images = "loaded later"
        images_root = args.images_root
    if images is None:
        logger.error("you should provide a way to find images")
        exit(1)

    faiss_index = args.load_faiss + ".faiss"
    faiss_meta = args.load_faiss + ".meta"
    index = pimmi.load_index(faiss_index, faiss_meta)

    if args.images_mining:
        images = pimmi.get_index_images(index, images_root)

    if args.nb_img:
        images = images.head(args.nb_img)

    logger.info("total number of queries " + str(len(images)))

    queries = tbx.split_pack(images, nb_per_split)

    glb_sub_result_df = None
    for pack in queries:
        pack_result_file = args.save_mining + "_" + str(pack[prm.dff_pack_id]).zfill(6) + ".csv"
        logger.info("query " + str(len(pack[prm.dff_pack_files])) + " files from pack " +
                    str(pack[prm.dff_pack_id]) + " -> " + pack_result_file)
        query_result, query_sub_result = pimmi.query_index_mt(index, pack[prm.dff_pack_files], images_root,
                                            pack=pack[prm.dff_pack_id], sub=args.sub)
        query_result = query_result.sort_values(by=[prm.dff_query_path, prm.dff_nb_match_ransac, prm.dff_ransac_ratio],
                                                ascending=False)
        query_result.to_csv(pack_result_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

        if glb_sub_result_df is None:
            glb_sub_result_df = query_sub_result
        else:
            glb_sub_result_df = pd.concat([glb_sub_result_df, query_sub_result])

    if args.sub:
        glb_sub_result_df.to_csv(args.save_mining + "_sub.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)