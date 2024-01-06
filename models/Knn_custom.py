import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        if self.distance_metric == 'euclidean':
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]
        elif self.distance_metric == 'cosine':
            distances = cdist([x], self.X_train, metric='cosine')[0]
        else:
            raise ValueError("Invalid distance metric. Supported metrics: 'euclidean', 'manhattan', 'cosine'.")

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
