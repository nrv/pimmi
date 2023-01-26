#!/bin/bash
set -e

IMAGES="/var/www/html/pimmi-copydays/copydays/"
VIRTUALENV=pimmi2
PYTHON_SCRIPT="scripts/copydays_groundtruth.py"
INDEX_NAME=copydays
FILE=benchmark_copydays.csv

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate $VIRTUALENV

mkdir -p index

if [ ! -f "$FILE" ]; then
    echo "real_time_fill,user_time_fill,sys_time_fill,\
real_time_query,user_time_query,sys_time_query,\
real_time_clusters,user_time_clusters,sys_time_clusters,\
index_type,sift_nfeatures,\
query_sift_knn,query_dist_ratio_threshold,\
algo,\
macro_p,macro_r,macro_f1,query_avg_p,\
name,image_count,\
sift_resize_image,sift_max_image_dimension,sift_nOctaveLayers,sift_contrastThreshold,sift_edgeThreshold,sift_sigma" > $FILE
fi

SIFT_RESIZE_IMAGE=true
SIFT_MAX_IMAGE_DIMENSION=512

for INDEX_TYPE in IVF1024,Flat OPQ16_64,IVF65536_HNSW32,PQ16

do  

    for SIFT_NFEATURES in 100 1000

    do 

        for SIFT_LAYERS in 2 3

        do

            for SIFT_EDGE in 5 100

            do
                
                for SIFT_CONTRAST in 0.08 0.12

                do
                    for SIFT_SIGMA in 1.4 1.8
                    do 
                        echo "********************Starting to fill index (nfeatures: $SIFT_NFEATURES ; layers: $SIFT_LAYERS ; edge: $SIFT_EDGE ; contrast: $SIFT_CONTRAST ; sigma: $SIFT_SIGMA)"
                        TIME_FILL=$(/usr/bin/time -f %e,%U,%S pimmi --sift-nOctaveLayers $SIFT_LAYERS --sift-contrastThreshold $SIFT_CONTRAST --sift-nfeatures $SIFT_NFEATURES --sift-edgeThreshold $SIFT_EDGE --sift-sigma $SIFT_SIGMA --index-type $INDEX_TYPE --silent true fill $IMAGES $INDEX_NAME --erase --force 2>&1)

                        for QUERY_SIFT_KNN in 100 200 1000 2000

                        do 

                            for QUERY_DIST in 0.2 0.4 0.6 0.8

                            do 
                                echo "********************Starting to query index (Nb knn: $QUERY_SIFT_KNN , query distance: $QUERY_DIST)"
                                TIME_QUERY=$(/usr/bin/time -f %e,%U,%S pimmi --query-sift-knn $QUERY_SIFT_KNN --query-dist-ratio-threshold $QUERY_DIST --index-type $INDEX_TYPE --silent true query $IMAGES $INDEX_NAME 2>&1)

                                for ALGO in components

                                do
                                    echo "********************Create clusters (Algo: $ALGO)"
                                    TIME_CLUSTERS=$(/usr/bin/time -f %e,%U,%S pimmi --index-type $INDEX_TYPE --algo $ALGO --silent true clusters $INDEX_NAME 2>&1 )

                                    echo "********************Evaluate"
                                    python $PYTHON_SCRIPT $IMAGES index/$INDEX_NAME.$INDEX_TYPE.mining.clusters.csv
                                    METRICS=$(pimmi eval index/$INDEX_NAME.$INDEX_TYPE.mining.groundtruth.csv --query-column image_status --csv)

                                    IMAGE_COUNT=`wc -l < index/$INDEX_NAME.$INDEX_TYPE.mining.groundtruth.csv`

                                    echo $METRICS

                                    echo "$TIME_FILL,$TIME_QUERY,$TIME_CLUSTERS,\"$INDEX_TYPE\",$SIFT_NFEATURES,$QUERY_SIFT_KNN,$QUERY_DIST,$ALGO,$METRICS,\
                                    $INDEX_NAME,$IMAGE_COUNT,$SIFT_RESIZE_IMAGE,$SIFT_MAX_IMAGE_DIMENSION,$SIFT_LAYERS,$SIFT_CONTRAST,$SIFT_EDGE,$SIFT_SIGMA" >> $FILE
                                    echo "********************"
                                    echo "********************"

                                done

                            done

                        done

                    done

                done

            done

        done  

    done

done