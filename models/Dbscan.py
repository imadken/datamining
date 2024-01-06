import numpy as np

class DBSCAN_custom:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.visited = None

    def fit(self, X):
        self.labels = np.zeros(len(X), dtype=int)
        self.visited = np.zeros(len(X), dtype=bool)

        cluster_id = 1  
        for i in range(len(X)):
            if not self.visited[i]:
                self.visited[i] = True
                neighbors = self._get_neighbors(X, i)

                if len(neighbors) < self.min_samples:
                    self.labels[i] = -1  #noise
                else:
                    self._expand_cluster(X, i, neighbors, cluster_id)
                    cluster_id += 1

    def _get_neighbors(self, X, i):
        distances = np.linalg.norm(X - X[i], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, i, neighbors, cluster_id):
        self.labels[i] = cluster_id

        for neighbor in neighbors:
            if not self.visited[neighbor]:
                self.visited[neighbor] = True
                new_neighbors = self._get_neighbors(X, neighbor)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))

            if self.labels[neighbor] == 0 or self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id
                
            