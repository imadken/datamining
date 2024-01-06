import numpy as np
from scipy.spatial.distance import cdist

class K_Means:
    def __init__(self, k=3, max_iters=100, tol=1e-4, distance_metric='euclidean'):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.distance_metric = distance_metric
        self.centroids = None

    def fit(self, X):
        # Initialize centroids 
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)

            # Update centroids based on the mean of points in each cluster
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        return self._assign_labels(X)

    def _assign_labels(self, X):
        if self.distance_metric == 'euclidean':
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        elif self.distance_metric == 'manhattan':
            distances = cdist(X, self.centroids, metric='cityblock')
        elif self.distance_metric == 'cosine':
            distances = cdist(X, self.centroids, metric='cosine')
        else:
            raise ValueError("Invalid distance metric. Supported metrics: 'euclidean', 'manhattan', 'cosine'.")

        return np.argmin(distances, axis=1)
