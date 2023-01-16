from fog.evaluation import best_matching_macro_average
import casanova
from collections import defaultdict


def read_file(file, truth_column, predicted_column, status_column=None, include_missing_predictions=True):
    truth = defaultdict(list)
    predicted = defaultdict(list)
    true_positives = defaultdict(dict)
    false_clusters = -1 # Create false clusters for images that are not in the clusters

    with open(file, "r") as f:
        reader = casanova.reader(f)
        truth_pos = reader.headers[truth_column]
        predicted_pos = reader.headers[predicted_column]
        status_pos = reader.headers[status_column] if status_column else None


        for i, row in enumerate(reader):

            if row[predicted_pos]:
                if int(row[predicted_pos]) < 0:
                    raise ValueError("Predicted clusters can't have negative values.")
                
                if status_pos:
                    if row[truth_pos] not in true_positives[row[predicted_pos]] and row[status_pos] == "original":
                        # Deal with query images. Requires the file to be sorted with query images at the top.
                        true_positives[row[predicted_pos]][row[truth_pos]] = {"true_positives_count": 0, "retrieved_count": 0}
                    
                    else:
                        # Ignore clusters that do not contain at least one query image
                        for cluster_id, clusters_stats in true_positives[row[predicted_pos]].items():
                            clusters_stats["retrieved_count"] += 1
                            if cluster_id == row[truth_pos]:
                                clusters_stats["true_positives_count"] += 1

                predicted[row[predicted_pos]].append(i)
                truth[row[truth_pos]].append(i)
                

            elif include_missing_predictions:
                
                if status_pos:
                # Make sure that precision = 0 for query images that are not in the clusters
                    if row[truth_pos] not in true_positives[row[predicted_pos]] and row[status_pos] == "original":
                        true_positives[row[predicted_pos]][row[truth_pos]] = {
                            "true_positives_count": 0, 
                            "retrieved_count": 1 # avoid division by 0
                            }

                # Images that have no cluster are considered to be the only image in their cluster
                predicted[false_clusters].append(i)
                truth[row[truth_pos]].append(i)

                false_clusters -= 1                    
    return truth.values(), predicted.values(), true_positives


def average_query_precision(true_positives):
    sum_of_precisions = 0
    number_of_queries = 0
    for query_images in true_positives.values():
        for stats in query_images.values():

            retrieved_count = stats["retrieved_count"]

            if retrieved_count != 0:
                precision = stats["true_positives_count"]/retrieved_count
            else:
                precision = 0

            sum_of_precisions += precision
            number_of_queries += 1

    if number_of_queries < 1:
        raise ValueError("true_positives is empty")
    return sum_of_precisions/number_of_queries


def evaluate(file, truth_column, predicted_column, status_column=None, include_missing_predictions=True):
    truth, predicted, true_positives = read_file(file, truth_column, predicted_column, status_column, include_missing_predictions)
    precision, recall, f1 = best_matching_macro_average(truth, predicted)
    if status_column:
        query_precision = average_query_precision(true_positives)
        return {"cluster precision": precision, "cluster recall": recall, "cluster f1": f1, "query average precision": query_precision}

    return {"cluster precision": precision, "cluster recall": recall, "cluster f1": f1}


if __name__ == "__main__":
    print(evaluate("/var/www/html/pimmi-copydays/index/copydays.IVF1024,Flat.mining.groundtruth.csv", "truth", "predicted", "image_status", False))