import math
import numpy as np
import cv2 as cv
import faiss
import pickle
import logging
import csv
import json
from json import JSONEncoder
import os.path
from multiprocessing import Queue, Process

from typing import List
from itertools import groupby

import pimmi.pimmi_parameters as constants
from pimmi.cli.config import parameters as prm
from pimmi.toolbox import Sift

grid_bits_per_dim = 10
grid_d = int(math.pow(2, grid_bits_per_dim))
grid_sz = grid_d * grid_d
grid_mask = grid_sz - 1
grid_shift = grid_bits_per_dim * 2
grid_x_mask = grid_d - 1
grid_aspect_ratio = True

dff_internal_meta = "meta"
dff_internal_id_generator = "id_generator"
dff_internal_pack_id = "pack_id"
dff_internal_query_result = "query_result"
dff_internal_faiss = "faiss"
dff_internal_faiss_type = "faiss_type"
dff_internal_faiss_nb_images = "faiss_nb_images"
dff_internal_faiss_nb_features = "faiss_nb_features"


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("pimmi")


def resize_if_needed(img):
    resized = False
    if prm.sift_resize_image:
        h, w = img.shape
        if h > prm.sift_max_image_dimension or w > prm.sift_max_image_dimension:
            r = prm.sift_max_image_dimension / max(h, w)
            img = cv.resize(img, (int(r * w), int(r * h)), interpolation=cv.INTER_AREA)
            resized = True
    return img, resized


def extract_sift_img(img, sift):
    kp, desc = sift.config.detectAndCompute(img, None)
    return kp, desc


def extract_sift(file, image_id, sift, pack=-1):
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    if img is None:
        if pack >= 0:
            pfx = str(pack) + " :: " + str(image_id)
        else:
            pfx = str(image_id)
        logger.error("~ [" + pfx + "] unable to read image file " + file)
        return None, None, None, None, None
    img, resized = resize_if_needed(img)
    kp, desc = extract_sift_img(img, sift)
    if image_id % 100 == 0:
        h, w = img.shape
        if pack >= 0:
            pfx = str(pack) + " :: " + str(image_id)
        else:
            pfx = str(image_id)
        logger.info(
            "~ [" + pfx + "] (" + str(prm.sift_nfeatures) + ", " + str(prm.sift_nOctaveLayers) + ", " +
            str(prm.sift_contrastThreshold) + ", " + str(prm.sift_edgeThreshold) + ", " + str(prm.sift_sigma)
            + ") extracting " + str(len(kp)) + " sift points for (" + str(w) + " x " +
            str(h) + ", " + str(resized) + ") " + file)
    ids = None
    if image_id is not None:
        ids = np.empty(len(kp), dtype="int64")
        for sift_point in range(0, len(kp)):
            ids[sift_point] = point_to_full_id(image_id, kp[sift_point].pt, img.shape[1], img.shape[0])
    return ids, kp, desc, img.shape[1], img.shape[0]


def point_to_full_id(image_id, pt, img_w, img_h):
    if grid_aspect_ratio:
        gx = int(math.floor(pt[0]))
        gy = int(math.floor(pt[1]))
    else:
        gx = int(math.floor(grid_d * pt[0] / img_w))
        gy = int(math.floor(grid_d * pt[1] / img_h))
    return (image_id << grid_shift) + grid_d * gy + gx


def full_id_to_image_id(full_id):
    return full_id >> grid_shift


def full_id_to_grid_id(full_id):
    return full_id & grid_mask


def grid_id_to_coord(grid_id):
    if not isinstance(grid_id, np.ndarray):
        grid_id = np.array(grid_id)
    gy = grid_id >> grid_bits_per_dim
    gx = grid_id & grid_x_mask
    points = np.vstack((gx, gy)).T
    return points.astype(int)


def full_id_to_coord(full_id):
    return grid_id_to_coord(full_id_to_grid_id(full_id))


def extract_sift_mt_function(task_queue, result_queue, sift):
    for task in iter(task_queue.get, constants.cst_stop):
        ids, kp, desc, width, height = extract_sift(task[constants.dff_image_path], task[constants.dff_image_id], sift)
        result_queue.put({constants.dff_image_id: task[constants.dff_image_id], constants.dff_ids: ids,
                          constants.dff_desc: desc, constants.dff_width: width, constants.dff_height: height})
    # logger.info("one thread has stopped")


def fill_index_mt(index, images, root_path, sift, only_empty_index=False):
    task_queue = Queue()
    result_queue = Queue()
    for i in range(prm.nb_threads):
        # logger.info("launching thread %d", i)
        Process(target=extract_sift_mt_function, args=(task_queue, result_queue, sift)).start()

    task_launched = 0
    for image_path in images:
        rel_image_path = os.path.relpath(image_path, root_path)
        index[dff_internal_meta][index[dff_internal_id_generator]] = {
            constants.dff_image_path: rel_image_path,
            constants.dff_width: None,
            constants.dff_height: None,
            constants.dff_nb_points: None,
            constants.dff_image_id: index[dff_internal_id_generator]
        }
        task_queue.put({
            constants.dff_image_path: image_path, 
            constants.dff_image_id: index[dff_internal_id_generator]})
        index[dff_internal_id_generator] = index[dff_internal_id_generator] + 1
        task_launched += 1
        if only_empty_index and task_launched >= prm.index_nb_images_to_train:
            break

    logger.info(
        "nb tasks launched : " + str(task_launched) + " / " + str(prm.index_nb_images_to_train))

    all_ids = None
    all_features = None
    for i in range(task_launched):
        result = result_queue.get()
        image_id = result[constants.dff_image_id]
        ids = result[constants.dff_ids]
        if ids is not None and len(ids) > 0:
            desc = result[constants.dff_desc]
            index[dff_internal_meta][image_id][constants.dff_width] = result[constants.dff_width]
            index[dff_internal_meta][image_id][constants.dff_height] = result[constants.dff_height]
            index[dff_internal_meta][image_id][constants.dff_nb_points] = len(ids)
            index[dff_internal_faiss_nb_images] = index[dff_internal_faiss_nb_images] + 1
            if image_id % 100 == 0:
                logger.info("~ [" + str(image_id) + "] result retrieved from queue : " + str(len(ids)))
            if all_features is None:
                all_features = []
                all_ids = []
            all_features.append(desc)
            all_ids.append(ids)
        if all_ids is not None and len(all_ids) >= prm.index_nb_images_to_train:
            all_features = np.vstack(all_features)
            all_ids = np.hstack(all_ids)
            if not index[dff_internal_faiss].is_trained:
                logger.info("---------- Training " + index[dff_internal_faiss_type] + " index on " + str(
                    len(all_ids)) + " features")
                index[dff_internal_faiss].train(all_features)
            if not only_empty_index:
                logger.info("---------- Adding " + str(len(all_ids)) + " features")
                index[dff_internal_faiss].add_with_ids(all_features, all_ids)
            index[dff_internal_faiss_nb_features] = index[dff_internal_faiss_nb_features] + len(all_ids)
            all_ids = None
            all_features = None

    for i in range(prm.nb_threads):
        task_queue.put(constants.cst_stop)

    if all_ids is not None:
        all_features = np.vstack(all_features)
        all_ids = np.hstack(all_ids)
        if not index[dff_internal_faiss].is_trained:
            logger.info("---------- Training " + index[dff_internal_faiss_type] + " index on " + str(
                len(all_ids)) + " features")
            index[dff_internal_faiss].train(all_features)
        if not only_empty_index:
            logger.info("---------- Adding " + str(len(all_ids)) + " features")
            index[dff_internal_faiss].add_with_ids(all_features, all_ids)
        index[dff_internal_faiss_nb_features] = index[dff_internal_faiss_nb_features] + len(all_ids)

    if not only_empty_index:
        # transform_meta_to_df(index)
        logger.info("index has " + str(index[dff_internal_faiss_nb_images]) + " images with " +
                    str(index[dff_internal_faiss_nb_features]) + " feature points")
    else:
        index[dff_internal_meta] = dict()
        index[dff_internal_faiss_nb_features] = 0
        index[dff_internal_faiss_nb_images] = 0

    return index


def create_index_mt(index_type, images, root_path, sift, only_empty_index=False, verbose=True):
    new_faiss = faiss.index_factory(128, index_type)
    new_faiss.verbose = verbose
    index = {dff_internal_faiss: new_faiss, dff_internal_faiss_type: index_type,
             dff_internal_meta: dict(),
             # dff_internal_meta_df: None,
             dff_internal_faiss_nb_features: 0, dff_internal_faiss_nb_images: 0,
             dff_internal_id_generator: 0}
    return fill_index_mt(index, images, root_path, sift, only_empty_index=only_empty_index)


def save_index(index, faiss_file, meta_file):
    faiss.write_index(index[dff_internal_faiss], faiss_file)
    if meta_file is not None:
        with open(meta_file, 'wb') as f:
            pickle.dump(index[dff_internal_meta], f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(index[dff_internal_faiss_type], f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(index[dff_internal_faiss_nb_images], f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(index[dff_internal_id_generator], f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(index[dff_internal_faiss_nb_features], f, pickle.HIGHEST_PROTOCOL)
        f.close()
        logger.info("index saved " + faiss_file + "  /  " + meta_file)
    else:
        logger.info("index saved without meta " + faiss_file)


def load_index(faiss_file, meta_file, correct=None, only_meta=False):
    index_faiss = None
    if not only_meta:
        index_faiss = faiss.read_index(faiss_file)
        logger.info("index loaded " + faiss_file)

    index = {dff_internal_faiss: index_faiss, dff_internal_meta: dict(),
             # dff_internal_meta_df: None,
             dff_internal_faiss_nb_images: 0, dff_internal_faiss_nb_features: 0}

    if meta_file is not None:
        if os.path.isfile(meta_file):
            with open(meta_file, 'rb') as f:
                index[dff_internal_meta] = pickle.load(f)
                if correct is None:
                    index[dff_internal_faiss_type] = pickle.load(f)
                    index[dff_internal_faiss_nb_images] = pickle.load(f)
                    index[dff_internal_id_generator] = pickle.load(f)
                    index[dff_internal_faiss_nb_features] = pickle.load(f)
                else:
                    index[dff_internal_faiss_type] = correct
                    index[dff_internal_faiss_nb_images] = 0
                    index[dff_internal_id_generator] = 0
                    index[dff_internal_faiss_nb_features] = 0
            f.close()
            # if len(index[dff_internal_meta]) > 0:
            #     transform_meta_to_df(index)
            logger.info("meta loaded " + meta_file)
    logger.info("index has " + str(index[dff_internal_faiss_nb_images]) + "/" + str(index[dff_internal_id_generator]) +
                " images with " + str(index[dff_internal_faiss_nb_features]) + " feature points")
    logger.info(" - type : " + index[dff_internal_faiss_type])
    return index


def get_index_images(index, path_prefix):
    # TODO simplify these 3 lines !
    all_path = [path_prefix + "/" + image[constants.dff_image_path] for image in index[dff_internal_meta].values()]
    all_image_ids = [image[constants.dff_image_id] for image in index[dff_internal_meta].values()]
    all_images = [{constants.dff_image_path: p, constants.dff_image_id: i} for p, i in zip(all_path, all_image_ids)]

    logger.info("found " + str(len(all_images)) + " images in index")
    return all_images


def query_index_extract_single_image(index, image_file, relative_image_path, sift, query_image_id=0, pack=-1):
    query_ids, query_kp, query_desc, query_width, query_height = extract_sift(
        image_file, query_image_id, sift, pack=pack
    )
    return query_index_single_image(
        index, query_ids, query_desc, query_width, query_height, relative_image_path, query_image_id
    )


class ImageResultEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class ImageResult:
    def __init__(self):
        self.keep: bool = True
        self.keep_smr: bool = True
        self.keep_smn: bool = True
        self.keep_rns: bool = True
        self.nb_match_total: int = -1
        self.nb_match_ransac: int = -1
        self.ransac_ratio: float = -1
        self.query_image_id: int = -1
        self.query_nb_points: int = -1
        self.query_path: str = ""
        self.query_width: int = -1
        self.query_height: int = -1
        self.result_image_id: int = -1
        self.result_nb_points: int = -1
        self.result_path: str = ""
        self.result_width: int = -1
        self.result_height: int = -1
        self.pack_id: int = -1

    def __repr__(self):
        return 'ir(' + str(self.result_image_id) + ')'

    def as_json(self):
        return ImageResultEncoder().encode(self)

    def as_dict(self):
        return self.__dict__


class ImageResultList:
    def __init__(self):
        self.__images: List[ImageResult] = []

    def __iter__(self):
        return iter(self.__images)

    def __len__(self):
        return len(self.__images)

    def __repr__(self):
        return 'irl[' + str(len(self.__images)) + ']'

    def add_image_result(self, r: ImageResult):
        self.__images.append(r)

    def add_new_image_result(self, query_nb_points, result_image_id, nb_match_total, query_image_id):
        r: ImageResult = ImageResult()
        r.keep = True
        r.query_nb_points = query_nb_points
        r.result_image_id = result_image_id.item()
        r.query_image_id = query_image_id
        r.nb_match_total = nb_match_total
        self.add_image_result(r)

    def add_new_image_results(self, query_nb_points, result_images, query_image_id):
        for result_image_id, nb_match_total in result_images.items():
            self.add_new_image_result(query_nb_points, result_image_id, nb_match_total, query_image_id)

    def filter_on_sift_match_ratio(self, min_nb_match):
        for image in self.__images:
            image.keep_smr = image.nb_match_total >= min_nb_match
            image.keep = image.keep and image.keep_smr

    def filter_on_sift_match_nb(self, min_nb_match):
        for image in self.__images:
            image.keep_smn = image.nb_match_total >= min_nb_match
            image.keep = image.keep and image.keep_smn

    def filter_on_ransac_sift_match_nb(self, min_nb_match):
        for image in self.__images:
            image.keep_rns = image.nb_match_ransac >= min_nb_match
            image.keep = image.keep and image.keep_rns

    def get_kept_image_ids(self):
        return set([image.result_image_id for image in self.__images if image.keep])

    def to_csv(self, file):
        with open(file, 'w', newline='') as csvfile:
            # 'keep', 'keep_smr', 'keep_smn', 'keep_rns',
            fieldnames = ['pack_id', 'query_image_id', 'result_image_id', 'query_path', 'result_path', 'nb_match_total',
                          'nb_match_ransac', 'ransac_ratio', 'query_nb_points', 'query_width', 'query_height',
                          'result_nb_points', 'result_width', 'result_height']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for i in self.__images:
                writer.writerow(i.as_dict())


class MatchedPoint:
    def __init__(self):
        self.query_id: int = -1
        self.image_id: int = -1
        self.point_id: int = -1

    def __repr__(self):
        return 'pr(' + str(self.image_id) + ')'


class MatchedPointList:
    def __init__(self):
        self.__points: List[MatchedPoint] = []

    def __repr__(self):
        return 'prl[' + str(len(self.__points)) + ']'

    def add_matched_point(self, r: MatchedPoint):
        self.__points.append(r)

    def add_new_matched_point(self, query_id, image_id, point_id):
        r = MatchedPoint()
        r.query_id = query_id
        r.image_id = image_id
        r.point_id = point_id
        self.add_matched_point(r)

    def add_new_matched_points(self, query_id, knn_ids):
        images = full_id_to_image_id(knn_ids)
        points = full_id_to_grid_id(knn_ids)
        for image_id, point_id in zip(images, points):
            self.add_new_matched_point(query_id, image_id, point_id)

    def keep_only_images(self, image_ids):
        kept_points = [point for point in self.__points if point.image_id in image_ids]
        self.__points = kept_points

    def group_by_image_id(self):
        self.__points.sort(key=lambda p: p.image_id)
        image_ids = []
        matched_points = []
        for image_id, points_iter in groupby(self.__points, lambda p: p.image_id):
            matched_points.append(list(points_iter))
            image_ids.append(image_id)
        return zip(image_ids, matched_points)


def query_index_single_image(index, sift_query_ids, query_desc, query_width, query_height, query_path, image_query_id):
    query_result: ImageResultList = ImageResultList()
    kept_matches: MatchedPointList = MatchedPointList()
    if len(sift_query_ids) > 0:
        result_images = dict()
        need_to_run_nn_again = True
        current_nn = prm.query_sift_knn
        max_hop = prm.query_adaptative_knn_max_step
        while need_to_run_nn_again and max_hop > 0:
            if prm.query_ransac:
                kept_matches = MatchedPointList()
            all_knn_dist, all_knn_ids = index[dff_internal_faiss].search(query_desc, current_nn)
            result_images = dict()
            current_nn_is_enough = True
            for sift_query_id, knn_ids, knn_dist in zip(sift_query_ids, all_knn_ids, all_knn_dist):
                filter_unknown_ids = knn_ids >= 0
                knn_ids = knn_ids[filter_unknown_ids]
                knn_dist = knn_dist[filter_unknown_ids]
                if prm.query_dist_filter and len(knn_ids) > 1:
                    filter_zero_dist = knn_dist == 0
                    knn_ids_zero = knn_ids[filter_zero_dist]
                    filter_nonzero_dist = ~filter_zero_dist
                    knn_ids_nonzero = knn_ids[filter_nonzero_dist]
                    knn_dist = knn_dist[filter_nonzero_dist]
                    if len(knn_dist) > 0:
                        first_non_zero_dist = knn_dist[0]
                        filter_on_dist_ratio = first_non_zero_dist / knn_dist > prm.query_dist_ratio_threshold
                        knn_ids_nonzero = knn_ids_nonzero[filter_on_dist_ratio]
                        knn_ids_filtered = np.concatenate((knn_ids_zero, knn_ids_nonzero))
                        if prm.query_adaptative_sift_knn and len(knn_ids_filtered) == len(knn_ids):
                            current_nn_is_enough = False
                            logger.info("    . " + str(sift_query_id) + " reached knn (" + str(current_nn) + ") limit "
                                        + str(knn_dist[-1]) + " / " + str(first_non_zero_dist))
                            break
                        else:
                            knn_ids = knn_ids_filtered

                if prm.query_ransac:
                    kept_matches.add_new_matched_points(sift_query_id, knn_ids)

                for result_image_id in np.unique(full_id_to_image_id(knn_ids)):
                    result_images[result_image_id] = result_images.get(result_image_id, 0) + 1

            if current_nn_is_enough:
                need_to_run_nn_again = False
            else:
                current_nn = current_nn * prm.query_adaptative_knn_mult
                max_hop -= 1

        query_result.add_new_image_results(len(sift_query_ids), result_images, image_query_id)

        if prm.query_match_ratio_filter:
            query_result.filter_on_sift_match_ratio(int(math.floor(len(sift_query_ids) * prm.sift_match_ratio_threshold)))

        if prm.query_match_nb_filter:
            query_result.filter_on_sift_match_nb(prm.query_match_nb_threshold)

        if prm.query_ransac:
            if prm.query_match_ratio_filter or prm.query_match_nb_filter:
                kept_matches.keep_only_images(query_result.get_kept_image_ids())

            ransac_result = {}

            for image_id, matched_pairs in kept_matches.group_by_image_id():
                query_points = grid_id_to_coord([mp.query_id for mp in matched_pairs]).reshape(-1, 1, 2)
                matched_points = grid_id_to_coord([mp.point_id for mp in matched_pairs]).reshape(-1, 1, 2)
                # ignored, mask = cv.findHomography(srcPoints=query_points, dstPoints=matched_points, method=cv.RANSAC,
                #                                   ransacReprojThreshold=5.0)
                # ignored, mask = cv.findEssentialMat(query_points, matched_points, method=cv.RANSAC, threshold=3.0,
                #                                     prob=0.99)
                ignored, mask = cv.estimateAffinePartial2D(query_points, matched_points)
                if mask is None:
                    nb_match_after_ransac = 0
                else:
                    nb_match_after_ransac = sum(mask.ravel().tolist())
                ransac_result[image_id] = nb_match_after_ransac

            for ir in query_result:
                ransac_ir = ransac_result.get(ir.result_image_id)
                if ransac_ir is not None:
                    ir.nb_match_ransac = ransac_ir
                else:
                    ir.nb_match_ransac = 0
                ir.ransac_ratio = ir.nb_match_ransac / ir.nb_match_total

                meta_ir = index[dff_internal_meta][ir.result_image_id]
                ir.result_path = meta_ir[constants.dff_image_path]
                ir.result_width = meta_ir[constants.dff_width]
                ir.result_height = meta_ir[constants.dff_height]
                ir.result_nb_points = meta_ir[constants.dff_nb_points]
                ir.query_path = query_path
                ir.query_width = query_width
                ir.query_height = query_height
                if prm.query_remove_from_results and ir.result_path == ir.query_path:
                    ir.keep = False

            if prm.query_match_nb_ransac_threshold:
                query_result.filter_on_ransac_sift_match_nb(prm.query_match_nb_ransac_threshold)
    return query_result


def query_index_mt_function(index, sift, task_queue, result_queue):
    for task in iter(task_queue.get, constants.cst_stop):
        query_result = query_index_extract_single_image(
            index,
            task[constants.dff_image_path],
            task[constants.dff_image_relative_path],
            sift,
            query_image_id=task[constants.dff_query_id],
            pack=task[dff_internal_pack_id]
        )
        result_queue.put(query_result)
    # logger.info("one thread has stopped")


def query_index_mt(index, images, root_path, pack=-1):
    task_queue = Queue()
    result_queue = Queue()

    sift = Sift(prm.sift_nfeatures, prm.sift_nOctaveLayers, prm.sift_contrastThreshold, prm.sift_edgeThreshold,
                    prm.sift_sigma, prm.nb_threads)

    for i in range(prm.nb_threads):
        # logger.info("launching thread %d", i)
        Process(target=query_index_mt_function, args=(index, sift, task_queue, result_queue)).start()

    task_launched = 0
    for image in images:
        image_path = image[constants.dff_image_path]
        if os.path.isfile(image_path):
            rel_image_path = os.path.relpath(image_path, root_path)
            task_queue.put(
                {
                    constants.dff_image_path: image_path,
                    constants.dff_image_relative_path: rel_image_path,
                    constants.dff_query_id: image[constants.dff_image_id],
                    dff_internal_pack_id: pack
                }
            )
            task_launched += 1
        else:
            logger.warning("unable to find image, will not query it : " + image_path)

    all_result = ImageResultList()
    for i in range(task_launched):
        result = result_queue.get()
        query_result: ImageResultList = result

        if query_result is None or len(query_result) == 0:
            continue
        for ir in query_result:
            if ir.keep:
                ir.pack_id = pack
                all_result.add_image_result(ir)

    for i in range(prm.nb_threads):
        task_queue.put(constants.cst_stop)

    return all_result
