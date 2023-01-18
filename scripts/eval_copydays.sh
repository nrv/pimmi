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


if [ ! -f "$FILE" ]; then
    echo "real_time_fill,user_time_fill,sys_time_fill,\
real_time_query,user_time_query,sys_time_query,\
real_time_clusters,user_time_clusters,sys_time_clusters,\
query_sift_knn,index_type,algo,\
macro_p,macro_r,macro_f1,query_avg_p,\
name,image_count" > $FILE
fi

for INDEX_TYPE in IVF1024,Flat OPQ16_64,IVF65536_HNSW32,PQ16

do

    echo "********************Starting to fill index (Index type: $INDEX_TYPE)"
    TIME_FILL=$(/usr/bin/time -f %e,%U,%S pimmi --index-type $INDEX_TYPE --silent true fill $IMAGES $INDEX_NAME --erase --force 2>&1 )

    for QUERY_SIFT_KNN in 10 20 100 200 1000

    do 
        echo "********************Starting to query index (Nb knn: $QUERY_SIFT_KNN)"
        TIME_QUERY=$(/usr/bin/time -f %e,%U,%S pimmi --query-sift-knn $QUERY_SIFT_KNN --index-type $INDEX_TYPE --silent true query $IMAGES $INDEX_NAME 2>&1)

        for ALGO in components louvain

        do
            echo "********************Create clusters (Algo: $ALGO)"
            TIME_CLUSTERS=$(/usr/bin/time -f %e,%U,%S pimmi --index-type $INDEX_TYPE --algo $ALGO --silent true clusters $INDEX_NAME 2>&1 )

            echo "********************Evaluate"
            python $PYTHON_SCRIPT $IMAGES index/$INDEX_NAME.$INDEX_TYPE.mining.clusters.csv
            METRICS=$(pimmi eval index/$INDEX_NAME.$INDEX_TYPE.mining.groundtruth.csv --query-column image_status --csv)

            IMAGE_COUNT=`wc -l < index/$INDEX_NAME.$INDEX_TYPE.mining.groundtruth.csv`

            echo $METRICS

            echo "$TIME_FILL,$TIME_QUERY,$TIME_CLUSTERS,$QUERY_SIFT_KNN,\"$INDEX_TYPE\",$ALGO,$METRICS,$INDEX_NAME,$IMAGE_COUNT" >> $FILE
            echo "********************"
            echo "********************"

        done

    done

done

