import math
import numpy as np
import pandas as pd
import cv2 as cv
import faiss
import pickle
import logging
import os.path
from multiprocessing import Queue, Process
import pimmi.pimmi_parameters as prm

grid_bits_per_dim = 10
grid_d = int(math.pow(2, grid_bits_per_dim))
grid_sz = grid_d * grid_d
grid_mask = grid_sz - 1
grid_shift = grid_bits_per_dim * 2
grid_x_mask = grid_d - 1

dff_internal_meta = "meta"
dff_internal_id_generator = "id_generator"
dff_internal_pack_id = "pack_id"
dff_internal_result_df = "result_df"
dff_internal_faiss = "faiss"
dff_internal_faiss_type = "faiss_type"
dff_internal_faiss_nb_images = "faiss_nb_images"
dff_internal_faiss_nb_features = "faiss_nb_features"
dff_internal_meta_df = "meta_df"

sift = None

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("pimmi")


def init_sift():
    global sift
    logger.info("Using opencv : " + cv.__version__)
    cv.setNumThreads(prm.nb_threads)
    sift = cv.SIFT_create(nfeatures=prm.sift_nfeatures, nOctaveLayers=prm.sift_nOctaveLayers,
                          contrastThreshold=prm.sift_contrastThreshold, edgeThreshold=prm.sift_edgeThreshold,
                          sigma=prm.sift_sigma)


def resize_if_needed(img):
    resized = False
    if prm.do_resize_image:
        h, w = img.shape
        if h > prm.max_image_dimension or w > prm.max_image_dimension:
            r = prm.max_image_dimension / max(h, w)
            img = cv.resize(img, (int(r * w), int(r * h)), interpolation=cv.INTER_AREA)
            resized = True
    return img, resized


def extract_sift_img(img):
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def extract_sift(file, image_id, pack=-1):
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    if img is None:
        if pack >= 0:
            pfx = str(pack) + " :: " + str(image_id)
        else:
            pfx = str(image_id)
        logger.error("~ [" + pfx + "] unable to read image file " + file)
        return None, None, None, None, None
    img, resized = resize_if_needed(img)
    kp, desc = extract_sift_img(img)
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
    if prm.do_preserve_aspect_ratio_for_quantized_coord:
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
    gy = grid_id >> grid_bits_per_dim
    gx = grid_id & grid_x_mask
    points = np.vstack((gx, gy)).T
    return points.astype(int)


def full_id_to_coord(full_id):
    return grid_id_to_coord(full_id_to_grid_id(full_id))


def as_point_df(query_id=None, image_id=None, point_id=None, knn_ids=None):
    if query_id is None:
        return pd.DataFrame(columns=[prm.dff_query_id, prm.dff_image_id, prm.dff_point_id])
    else:
        if image_id is not None and point_id is not None:
            return pd.DataFrame([[query_id, image_id, point_id]],
                                columns=[prm.dff_query_id, prm.dff_image_id, prm.dff_point_id])
        elif knn_ids is not None:
            images = full_id_to_image_id(knn_ids)
            points = full_id_to_grid_id(knn_ids)
            return pd.DataFrame({
                prm.dff_query_id: query_id,
                prm.dff_image_id: images,
                prm.dff_point_id: points
            })
        else:
            return None


def extract_sift_mt_function(task_queue, result_queue):
    for task in iter(task_queue.get, prm.cst_stop):
        ids, kp, desc, width, height = extract_sift(task[prm.dff_image_path], task[prm.dff_image_id])
        result_queue.put({prm.dff_image_id: task[prm.dff_image_id], prm.dff_ids: ids, prm.dff_desc: desc,
                          prm.dff_width: width, prm.dff_height: height})
    # logger.info("one thread has stopped")


def fill_index_mt(index, images, root_path, only_empty_index=False):
    init_sift()

    task_queue = Queue()
    result_queue = Queue()
    for i in range(prm.nb_threads):
        # logger.info("launching thread %d", i)
        Process(target=extract_sift_mt_function, args=(task_queue, result_queue)).start()

    task_launched = 0
    for image_path in images:
        rel_image_path = os.path.relpath(image_path, root_path)
        index[dff_internal_meta][index[dff_internal_id_generator]] = {prm.dff_image_path: rel_image_path,
                                                                     prm.dff_width: None,
                                                                     prm.dff_height: None,
                                                                     prm.dff_nb_points: None,
                                                                     prm.dff_image_id: index[dff_internal_id_generator]}
        task_queue.put({prm.dff_image_path: image_path, prm.dff_image_id: index[dff_internal_id_generator]})
        index[dff_internal_id_generator] = index[dff_internal_id_generator] + 1
        task_launched += 1
        if only_empty_index and task_launched >= prm.nb_images_to_train_index:
            break

    logger.info(
        "nb tasks launched : " + str(task_launched) + " / " + str(prm.nb_images_to_train_index))

    all_ids = None
    all_features = None
    for i in range(task_launched):
        result = result_queue.get()
        image_id = result[prm.dff_image_id]
        ids = result[prm.dff_ids]
        if ids is not None and len(ids) > 0:
            desc = result[prm.dff_desc]
            index[dff_internal_meta][image_id][prm.dff_width] = result[prm.dff_width]
            index[dff_internal_meta][image_id][prm.dff_height] = result[prm.dff_height]
            index[dff_internal_meta][image_id][prm.dff_nb_points] = len(ids)
            index[dff_internal_faiss_nb_images] = index[dff_internal_faiss_nb_images] + 1
            if image_id % 100 == 0:
                logger.info("~ [" + str(image_id) + "] result retrieved from queue : " + str(len(ids)))
            if all_features is None:
                all_features = []
                all_ids = []
            all_features.append(desc)
            all_ids.append(ids)
        if all_ids is not None and len(all_ids) >= prm.nb_images_to_train_index:
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
        task_queue.put(prm.cst_stop)

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
        transform_meta_to_df(index)
        logger.info("index has " + str(index[dff_internal_faiss_nb_images]) + " images with " +
                    str(index[dff_internal_faiss_nb_features]) + " feature points")
    else:
        index[dff_internal_meta] = dict()
        index[dff_internal_faiss_nb_features] = 0
        index[dff_internal_faiss_nb_images] = 0

    return index


def create_index_mt(index_type, images, root_path, only_empty_index=False):
    new_faiss = faiss.index_factory(128, index_type)
    new_faiss.verbose = True
    index = {dff_internal_faiss: new_faiss, dff_internal_faiss_type: index_type,
             dff_internal_meta: dict(), dff_internal_meta_df: None,
             dff_internal_faiss_nb_features: 0, dff_internal_faiss_nb_images: 0,
             dff_internal_id_generator: 0}
    return fill_index_mt(index, images, root_path, only_empty_index=only_empty_index)


# TODO check these 2 methods, patched too hardly
def meta_as_df(meta_json):
    index_meta_df = pd.DataFrame(meta_json.items(), columns=["tmp_id", "tmp_stuff"])
    unnested = pd.json_normalize(index_meta_df["tmp_stuff"])
    return index_meta_df.join(unnested).drop(columns=["tmp_id", "tmp_stuff"])


def transform_meta_to_df(index):
    index[dff_internal_meta_df] = meta_as_df(index[dff_internal_meta])
    # index[dff_internal_meta_df][prm.dff_image_path] = "/" + index[dff_internal_meta_df][prm.dff_image_path]


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

    index = {dff_internal_faiss: index_faiss, dff_internal_meta: dict(), dff_internal_meta_df: None,
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
            if len(index[dff_internal_meta]) > 0:
                transform_meta_to_df(index)
            logger.info("meta loaded " + meta_file)
    logger.info("index has " + str(index[dff_internal_faiss_nb_images]) + "/" + str(index[dff_internal_id_generator]) +
                " images with " + str(index[dff_internal_faiss_nb_features]) + " feature points")
    logger.info(" - type : " + index[dff_internal_faiss_type])
    return index


def get_index_images(index, path_prefix):
    all_path = path_prefix + "/" + index[dff_internal_meta_df][prm.dff_image_path]
    all_image_ids = index[dff_internal_meta_df][prm.dff_image_id]
    all_images = pd.DataFrame({
        prm.dff_image_path: all_path,
        prm.dff_image_id: all_image_ids
    })
    logger.info("found " + str(len(all_images)) + " images in index")
    return all_images


def query_index_extract_single_image(index, image_file, relative_image_path, query_id=0, pack=-1):
    query_ids, query_kp, query_desc, query_width, query_height = extract_sift(image_file, query_id, pack=pack)
    return query_index_single_image(index, query_ids, query_desc, query_width, query_height, relative_image_path)


def query_index_single_image(index, query_ids, query_desc, query_width, query_height, query_path):
    result_df = pd.DataFrame(
        columns=[prm.dff_keep, prm.dff_query_nb_points, prm.dff_result_image, prm.dff_nb_match_total])
    if len(query_ids) > 0:
        need_to_run_nn_again = True
        current_nn = prm.each_sift_nn
        max_hop = 3
        while need_to_run_nn_again and max_hop > 0:
            if prm.do_ransac:
                kept_matches = as_point_df()
            all_knn_dist, all_knn_ids = index[dff_internal_faiss].search(query_desc, current_nn)
            result_images = dict()
            current_nn_is_enough = True
            for query_id, knn_ids, knn_dist in zip(query_ids, all_knn_ids, all_knn_dist):
                if prm.do_filter_on_sift_dist:
                    filter_zero_dist = knn_dist == 0
                    knn_ids_zero = knn_ids[filter_zero_dist]
                    filter_nonzero_dist = ~filter_zero_dist
                    knn_ids_nonzero = knn_ids[filter_nonzero_dist]
                    knn_dist = knn_dist[filter_nonzero_dist]
                    first_non_zero_dist = knn_dist[0]
                    filter_on_dist_ratio = first_non_zero_dist / knn_dist > prm.sift_dist_ratio_threshold
                    knn_ids_nonzero = knn_ids_nonzero[filter_on_dist_ratio]
                    knn_ids_filtered = np.concatenate((knn_ids_zero, knn_ids_nonzero))
                    if prm.adaptative_sift_nn and len(knn_ids_filtered) == len(knn_ids):
                        current_nn_is_enough = False
                        logger.info("    . " + str(query_id) + " reached knn (" + str(current_nn) + ") limit "
                                    + str(knn_dist[-1]) + " / " + str(first_non_zero_dist))
                        break
                    else:
                        knn_ids = knn_ids_filtered

                if prm.do_ransac:
                    kept_matches = pd.concat([kept_matches, as_point_df(query_id=query_id, knn_ids=knn_ids)])

                for result_image_id in np.unique(full_id_to_image_id(knn_ids)):
                    result_images[result_image_id] = result_images.get(result_image_id, 0) + 1

            if current_nn_is_enough:
                need_to_run_nn_again = False
            else:
                current_nn = current_nn * 4
                max_hop -= 1

        result_df = pd.DataFrame({
            prm.dff_keep: True,
            prm.dff_query_nb_points: len(query_ids),
            prm.dff_result_image: np.fromiter(result_images.keys(), dtype=int),
            prm.dff_nb_match_total: np.fromiter(result_images.values(), dtype=int)
        })

        if prm.do_filter_on_sift_match_ratio:
            min_nb_match = int(math.floor(len(query_ids) * prm.sift_match_ratio_threshold))
            result_df[prm.dff_keep_smr] = result_df[prm.dff_nb_match_total] >= min_nb_match
            result_df[prm.dff_keep] = result_df[[prm.dff_keep, prm.dff_keep_smr]].all(axis='columns')

        if prm.do_filter_on_sift_match_nb:
            result_df[prm.dff_keep_smn] = result_df[prm.dff_nb_match_total] >= prm.sift_match_nb_threshold
            result_df[prm.dff_keep] = result_df[[prm.dff_keep, prm.dff_keep_smn]].all(axis='columns')

        if prm.do_ransac:
            if prm.do_filter_on_sift_match_ratio or prm.do_filter_on_sift_match_nb:
                kept_matches = kept_matches[
                    kept_matches[prm.dff_image_id].isin(result_df[result_df[prm.dff_keep]][prm.dff_result_image])]
            ransac_df = pd.DataFrame({prm.dff_result_image: pd.Series(dtype='int'),
                                      prm.dff_nb_match_ransac: pd.Series(dtype='int'),
                                      prm.dff_keep_rns: pd.Series(dtype='bool')})

            for matched_image in kept_matches.groupby(prm.dff_image_id):
                matches = matched_image[1]
                query_points = grid_id_to_coord(matches[prm.dff_query_id].values).reshape(-1, 1, 2)
                matched_points = grid_id_to_coord(matches[prm.dff_point_id].values).reshape(-1, 1, 2)
                # ignored, mask = cv.findHomography(srcPoints=query_points, dstPoints=matched_points, method=cv.RANSAC,
                #                                   ransacReprojThreshold=5.0)
                # ignored, mask = cv.findEssentialMat(query_points, matched_points, method=cv.RANSAC, threshold=3.0,
                #                                     prob=0.99)
                ignored, mask = cv.estimateAffinePartial2D(query_points, matched_points)
                if mask is None:
                    nb_match_after_ransac = 0
                else:
                    nb_match_after_ransac = sum(mask.ravel().tolist())
                ransac_df = pd.concat([ransac_df,
                                       pd.DataFrame([[matched_image[0], nb_match_after_ransac,
                                                      nb_match_after_ransac >= prm.sift_match_nb_after_ransac_threshold]],
                                                    columns=[prm.dff_result_image, prm.dff_nb_match_ransac,
                                                             prm.dff_keep_rns])])

            result_df = result_df.join(ransac_df.set_index(prm.dff_result_image), on=prm.dff_result_image)
            result_df[prm.dff_nb_match_ransac] = result_df[prm.dff_nb_match_ransac].fillna(0)
            result_df[prm.dff_ransac_ratio] = result_df[prm.dff_nb_match_ransac] / result_df[prm.dff_nb_match_total]
            result_df[prm.dff_keep] = result_df[[prm.dff_keep, prm.dff_keep_rns]].all(axis="columns")

        result_df = pd.merge(result_df, index[dff_internal_meta_df], how="inner", left_on=prm.dff_result_image,
                             right_on=prm.dff_image_id)
        result_df = result_df.drop(columns=prm.dff_image_id).rename(columns={
            prm.dff_image_path: prm.dff_result_path,
            prm.dff_width: prm.dff_result_width,
            prm.dff_height: prm.dff_result_height,
            prm.dff_nb_points: prm.dff_result_nb_points
        })
        result_df[prm.dff_query_path] = query_path
        result_df[prm.dff_query_width] = query_width
        result_df[prm.dff_query_height] = query_height
        result_df = pd.merge(result_df, index[dff_internal_meta_df][[prm.dff_image_path, prm.dff_image_id]], how="left",
                             left_on=prm.dff_query_path, right_on=prm.dff_image_path)
        result_df = result_df.drop(columns=prm.dff_image_path).rename(columns={
            prm.dff_image_id: prm.dff_query_image
        })

        if prm.remove_query_from_results:
            result_df.loc[result_df[prm.dff_result_path] == query_path, prm.dff_keep] = False
    return result_df


def query_index_mt_function(index, task_queue, result_queue):
    for task in iter(task_queue.get, prm.cst_stop):
        result_df = query_index_extract_single_image(index, task[prm.dff_image_path], task[prm.dff_image_relative_path],
                                                     task[prm.dff_query_id], pack=task[dff_internal_pack_id])
        result_queue.put({prm.dff_query_id: task[prm.dff_query_id], dff_internal_result_df: result_df})
    # logger.info("one thread has stopped")


def query_index_mt(index, images, root_path, pack=-1):
    init_sift()

    task_queue = Queue()
    result_queue = Queue()
    for i in range(prm.nb_threads):
        # logger.info("launching thread %d", i)
        Process(target=query_index_mt_function, args=(index, task_queue, result_queue)).start()

    task_launched = 0
    for row, image in images.iterrows():
        image_path = image[prm.dff_image_path]
        rel_image_path = os.path.relpath(image_path, root_path)
        task_queue.put({prm.dff_image_path: image_path, prm.dff_image_relative_path: rel_image_path,
                        prm.dff_query_id: image[prm.dff_image_id], dff_internal_pack_id: pack})
        task_launched += 1

    all_result = []
    for i in range(task_launched):
        result = result_queue.get()
        result_df = result[dff_internal_result_df]

        if result_df is None or len(result_df) == 0:
            continue
        result_df = result_df[result_df[prm.dff_keep]]
        if result_df is None or len(result_df) == 0:
            continue
        all_result.append(result_df)

    logger.info("~ concatenating results")
    all_result = pd.concat(all_result)

    if all_result is not None:
        all_result[dff_internal_pack_id] = pack
    for i in range(prm.nb_threads):
        task_queue.put(prm.cst_stop)

    return all_result
