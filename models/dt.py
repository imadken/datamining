import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)

        # If all samples belong to the same class or max depth reached, return a leaf node
        if len(unique_classes) == 1 or (
            self.max_depth is not None and depth == self.max_depth
        ):
            return {"class": unique_classes[0], "count": counts[0]}

        # If no features left, return a leaf node with the majority class
        if num_features == 0:
            majority_class = unique_classes[np.argmax(counts)]
            return {"class": majority_class, "count": counts[np.argmax(counts)]}

        
        best_attribute, best_threshold = self._find_best_split(X, y)

        # If no useful split found, return a leaf node with the majority class
        if best_attribute is None:
            majority_class = unique_classes[np.argmax(counts)]
            return {"class": majority_class, "count": counts[np.argmax(counts)]}

        # Split the data based on the best attribute and threshold
        left_mask = X[:, best_attribute] <= best_threshold
        right_mask = ~left_mask

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return the decision node
        return {
            "attribute": best_attribute,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_attribute = None
        best_threshold = None
        best_information_gain = 0

        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    information_gain = self._calculate_information_gain(
                        y, left_mask, right_mask
                    )

                    if information_gain > best_information_gain:
                        best_information_gain = information_gain
                        best_attribute = feature
                        best_threshold = threshold

        return best_attribute, best_threshold

    def _calculate_information_gain(self, y, left_mask, right_mask):
        total_entropy = self._calculate_entropy(y)

        left_entropy = self._calculate_entropy(y[left_mask])
        right_entropy = self._calculate_entropy(y[right_mask])

        num_left = np.sum(left_mask)
        num_right = np.sum(right_mask)
        total_samples = num_left + num_right

        information_gain = total_entropy - (
            (num_left / total_samples) * left_entropy
            + (num_right / total_samples) * right_entropy
        )
        return information_gain

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def predict(self, X):
        predictions = np.array([self._predict_sample(x, self.tree) for x in X])
        return predictions

    def _predict_sample(self, x, node):
        if "class" in node:
            return node["class"]
        else:
            if x[node["attribute"]] <= node["threshold"]:
                return self._predict_sample(x, node["left"])
            else:
                return self._predict_sample(x, node["right"])

    def get_params(self, deep=True):
        return {"max_depth": self.max_depth}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self