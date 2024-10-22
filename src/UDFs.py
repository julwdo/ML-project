import numpy as np

def scaled_entropy(y):
        probabilities = np.bincount(y) / len(y)
        scaled_entropy_value = -np.sum(probabilities * np.log2(probabilities))
        return scaled_entropy_value

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class Tree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        
    def fit(self, X, y):
        self.n_features = min(self.n_features or X.shape[1], X.shape[1])
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (n_samples < self.min_samples_split or depth >= self.max_depth or n_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        features = np.random.choice(n_features, self.n_features, replace = False)

        best_feature, best_threshold = self._best_criterion(X, y, features)

        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts = True)
        most_common = unique[np.argmax(counts)]
        
        return most_common
    
    def _best_criterion(self, X, y, features):
        best_gain = -1
        split_feature, split_threshold = None, None
        
        for feature in features:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature
                    split_threshold = threshold

        return split_feature, split_threshold

    def _information_gain(self, y, X_column, threshold):
        h_current = scaled_entropy(y)

        left_indices, right_indices = self._split(X_column, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        h_left, h_right = scaled_entropy(y[left_indices]), scaled_entropy(y[right_indices])
        h_children = (n_left / n) * h_left + (n_right / n) * h_right
        
        h_difference = h_current - h_children

        return h_difference
    
    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        
        return left_indices, right_indices
    
    def predict(self, X):
        return [self._traverse_tree(x) for x in X]

    def _traverse_tree(self, x):
        node = self.root
        
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
                
        return node.value