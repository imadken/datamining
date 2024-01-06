import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score


def confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1

    return cm

def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    correct_predictions = np.sum(np.diag(cm))
    total_predictions = np.sum(cm)
    acc = correct_predictions / total_predictions
    return acc

def precision(y_true, y_pred, class_label):
    cm = confusion_matrix(y_true, y_pred)
    true_positive = cm[class_label, class_label]
    false_positive = np.sum(cm[:, class_label]) - true_positive
    precision = true_positive / (true_positive + false_positive)
    return precision

def recall(y_true, y_pred, class_label):
    cm = confusion_matrix(y_true, y_pred)
    true_positive = cm[class_label, class_label]
    false_negative = np.sum(cm[class_label, :]) - true_positive
    recall = true_positive / (true_positive + false_negative)
    return recall

def f1_score(y_true, y_pred, class_label):
    prec = precision(y_true, y_pred, class_label)
    rec = recall(y_true, y_pred, class_label)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

def specificity(y_true, y_pred, class_label):
    cm = confusion_matrix(y_true, y_pred)
    true_negative = np.sum(np.diag(cm)) - cm[class_label, class_label]
    false_positive = np.sum(cm[:, class_label]) - cm[class_label, class_label]
    spec = true_negative / (true_negative + false_positive)
    return spec

def calculate_metrics(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / len(y_true)
    specificity = (true_negative) / (true_negative+false_positive)

    return precision, recall, f1_score, accuracy,specificity

def calculate_intra_cluster_distance(data, labels):
    unique_labels = np.unique(labels)
    intra_cluster_distances = []

    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            pairwise_distances = np.linalg.norm(cluster_points - cluster_points[:, np.newaxis], axis=-1)
            avg_distance = np.sum(pairwise_distances) / (len(cluster_points) * (len(cluster_points) - 1))
            intra_cluster_distances.append(avg_distance)

    return np.mean(intra_cluster_distances)

def calculate_inter_cluster_distance(data, labels):
    unique_labels = np.unique(labels)
    centroids = np.array([np.mean(data[labels == label], axis=0) for label in unique_labels])
    pairwise_distances = np.linalg.norm(centroids - centroids[:, np.newaxis], axis=-1)
    return np.mean(pairwise_distances)

def calculate_silhouette_score(data, labels):
    data = np.array(data)  # Ensure data is a NumPy array
    n = len(data)
    distances = cdist(data, data, metric='euclidean')
    silhouette_scores = []

    for i in range(n):
        a_i = np.mean([distances[i, j] for j in range(n) if labels[j] == labels[i] and j != i])
        b_i = min([np.mean([distances[i, j] for j in range(n) if labels[j] == k]) for k in set(labels) if k != labels[i]])
        silhouette_scores.append((b_i - a_i) / max(a_i, b_i))

    return np.mean(silhouette_scores)


def silhouette(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg

def davies_bouldin_index(X, labels, centroids):
    num_clusters = len(centroids)
    
    if num_clusters <= 1:
        return 0.0  # Davies-Bouldin index is 0 when there is only one cluster

    sigma = np.zeros(num_clusters)

    for i in range(num_clusters):
        cluster_points = X[labels == i]

        if len(cluster_points) == 0:
            # Skip empty clusters
            continue

        mean_i = centroids[i]
        distances_i = pairwise_distances(cluster_points, [mean_i])
        sigma[i] = np.mean(distances_i)

    result = 0.0
    for i in range(num_clusters):
        max_similarity = 0.0

        for j in range(num_clusters):
            if i != j:
                similarity = (sigma[i] + sigma[j]) / pairwise_distances([centroids[i]], [centroids[j]])
                if similarity > max_similarity:
                    max_similarity = similarity

        result += max_similarity

    return result / num_clusters

def inertia(X, labels, centroids):
    total_inertia = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        inertia_i = np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2)
        total_inertia += inertia_i
    return total_inertia