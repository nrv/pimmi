# Some parameters that may be overridden
nb_threads = 8
do_resize_image = True
max_image_dimension = 512
do_preserve_aspect_ratio_for_quantized_coord = True
nb_images_to_train_index = 25000
each_sift_nn = 1000
do_filter_on_sift_dist = True
sift_dist_ratio_threshold = 0.6
adaptative_sift_nn = True
do_filter_on_sift_match_ratio = False
sift_match_ratio_threshold = 0.01
do_filter_on_sift_match_nb = True
sift_match_nb_threshold = 5
sift_match_nb_after_ransac_threshold = 5
do_ransac = True
remove_query_from_results = True

sift_nfeatures = 1000
sift_nOctaveLayers = 1
sift_contrastThreshold = 0.1
sift_edgeThreshold = 10
sift_sigma = 1.6

# dataframe fields
dff_query_id = "query_id"
dff_image_id = "image_id"
dff_twitter_id = "twitter_id"
dff_point_id = "point_id"
dff_image_path = "path"
dff_image_relative_path = "relative_path"
dff_ids = "ids"
dff_desc = "desc"
dff_width = "width"
dff_height = "height"
dff_keep = "keep"
dff_nb_points = "nb_points"
dff_nb_match_total = "nb_match_total"
dff_nb_match_ransac = "nb_match_ransac"
dff_ransac_ratio = "ransac_ratio"
dff_keep_smr = dff_keep + "_smr"
dff_keep_smn = dff_keep + "_smn"
dff_keep_rns = dff_keep + "_rns"
dff_query_pfx = "query_"
dff_result_pfx = "result_"
dff_result_nb_points = dff_result_pfx + dff_nb_points
dff_result_image = dff_result_pfx + dff_image_id
dff_result_path = dff_result_pfx + dff_image_path
dff_result_width = dff_result_pfx + dff_width
dff_result_height = dff_result_pfx + dff_height
dff_query_nb_points = dff_query_pfx + dff_nb_points
dff_query_image = dff_query_pfx + dff_image_id
dff_query_path = dff_query_pfx + dff_image_path
dff_query_width = dff_query_pfx + dff_width
dff_query_height = dff_query_pfx + dff_height
dff_pack_id = "id"
dff_pack_files = "files"

mex_id = "id"
mex_relative_path = "relativePath"
mex_ext_retrieved = "extRetrieved"
mex_ext_unknown_id = "extIsUnknownId"
mex_ext_locked_id = "extIsLockedId"
mex_ext_over_capacity = "extIsOverCapacity"
mex_ext_need_retrieve = "extNeedToRetrieve"
mex_ext_file_type = "fileType"
mex_ext_media_url = "mediaUrl"
mex_ext_first_seen = "firstSeen"
mex_ext_at_least_one_french = "atLeastOneFrench"
mex_nbSeen = "nbSeen"
mex_sha256 = "sha256"

# other string constants
cst_stop = "STOP"
