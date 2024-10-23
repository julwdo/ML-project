import numpy as np
import pandas as pd

def compute_entropy(y):
    """Calculate the entropy of labels.

    Args:
        y (np.ndarray): Array of class labels.

    Returns:
        float: Entropy of the class labels.
    """
    probabilities = np.bincount(y) / len(y)
    return -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))

def compute_gini(y):
    """Calculate the Gini impurity of labels.

    Args:
        y (np.ndarray): Array of class labels.

    Returns:
        float: Gini impurity of the class labels.
    """
    probabilities = np.bincount(y) / len(y)
    return 1.0 - np.sum(probabilities ** 2)

def compute_accuracy(y, y_predicted):
    """Compute accuracy as the percentage of correct predictions.

    Args:
        y (np.ndarray): True class labels.
        y_predicted (np.ndarray): Predicted class labels.

    Returns:
        float: Accuracy of the predictions.
    """
    return np.sum(y == y_predicted) / len(y)

class TreeNode:
    def __init__(self, feature_index=None, threshold_value=None, left_child=None, right_child=None, leaf_value=None):
        """Initialize a tree node.

        Args:
            feature_index (int, optional): Index of the feature used for splitting.
            threshold_value (float, optional): Threshold value for the split.
            left_child (TreeNode, optional): Left child node.
            right_child (TreeNode, optional): Right child node.
            leaf_value (int, optional): Class label if the node is a leaf.
        """
        self.feature_index = feature_index
        self.threshold_value = threshold_value
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_value = leaf_value

    def is_leaf(self):
        """Check if the node is a leaf.

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return self.leaf_value is not None
    
class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=None, n_features=None, 
                 criterion="entropy", min_information_gain=0.0, n_quantiles=None, 
                 one_vs_rest=False):
        """Initialize the Decision Tree Classifier.

        Args:
            min_samples_split (int): Minimum number of samples required to split an internal node.
            max_depth (int, optional): Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            n_features (int, optional): Number of features to consider when looking for the best split.
            criterion (str): Function to measure the quality of a split ("entropy" or "gini").
            min_information_gain (float): Minimum information gain required to make a split.
            n_quantiles (int, optional): Specifies the number of quantiles to use for creating 
                threshold values when splitting using numerical features. If set to None, the classifier 
                will use midpoints between all unique values in the feature. If an integer is provided, 
                the classifier will create quantile thresholds based on the specified number of equally 
                spaced quantiles, which can help manage the number of splits in features with a large 
                number of unique values. Defaults to None.
            one_vs_rest (bool, optional): If True, uses a one-vs-rest strategy for string categorical
            features, allowing for binary splits where one category is isolated on one side, and all
            other categories are grouped on the other side. If False, categorical splits are performed
            such that all values alphabetically lower than the threshold (including the threshold) go
            to the left node, while the remaining values go to the right node. Defaults to False.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.min_information_gain = min_information_gain
        self.n_quantiles = n_quantiles
        self.one_vs_rest = one_vs_rest
        self.root = None
        
    def fit(self, X, y):
        """Fit the model to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Class labels.
        """
        print("Fitting the model...")
        self.n_features = min(self.n_features or X.shape[1], X.shape[1])
        self.root = self._build_tree(X, y)
        print("Model fitting completed.")

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Class labels.
            depth (int): Current depth in the tree.

        Returns:
            TreeNode: Root node of the constructed subtree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stop conditions
        if (n_samples < self.min_samples_split or 
            (self.max_depth is not None and depth >= self.max_depth) or 
            n_labels == 1):
            most_common_label = self._get_most_common_label(y)
            return TreeNode(leaf_value=most_common_label)

        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
        best_feature_index, best_threshold_value = self._find_best_split(X, y, feature_indices)
        
        # Avoid invalid splits
        if best_feature_index is None or best_threshold_value is None:
            most_common_label = self._get_most_common_label(y)
            return TreeNode(leaf_value=most_common_label)
        
        left_indices, right_indices = self._split(X[:, best_feature_index], best_threshold_value)

        # Create child nodes
        left_child = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)
        
        return TreeNode(best_feature_index, best_threshold_value, left_child, right_child)
    
    def _get_most_common_label(self, y):
        """Find the most common label in the labels array.

        Args:
            y (np.ndarray): Class labels.

        Returns:
            int: Most common class label.
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        return most_common_label
    
    def _find_best_split(self, X, y, feature_indices):
        """Determine the best feature and threshold for splitting.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Class labels.
            feature_indices (np.ndarray): Randomly selected feature indices.

        Returns:
            tuple: Best feature index and threshold value for splitting.
        """
        best_gain = self.min_information_gain
        best_feature_index, best_threshold_value = None, None
        
        for feature_index in feature_indices:
            feature_column = X[:, feature_index]
            
            if pd.api.types.is_string_dtype(feature_column):
                unique_thresholds = np.unique(feature_column)
            else:
                unique_values = np.unique(feature_column)
                
                if self.n_quantiles is None:
                    unique_thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                else:
                    unique_thresholds = np.quantile(unique_values, np.linspace(0, 1, self.n_quantiles + 1)[1:-1])
            
            for threshold_value in unique_thresholds:
                gain = self._calculate_information_gain(y, feature_column, threshold_value)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold_value = threshold_value

        return best_feature_index, best_threshold_value

    def _calculate_information_gain(self, y, feature_column, threshold_value):
        """Calculate the information gain from a split based on the selected criterion.

        Args:
            y (np.ndarray): Class labels.
            feature_column (np.ndarray): Feature values for the selected feature.
            threshold_value (float or str): Threshold value for the split.

        Returns:
            float: Information gain from the split.
        """
        impurity_functions = {
            "gini": compute_gini,
            "entropy": compute_entropy
        }
        impurity_before_split = impurity_functions[self.criterion](y)

        left_indices, right_indices = self._split(feature_column, threshold_value)

        # Avoid invalid splits
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        impurity_left = impurity_functions[self.criterion](y[left_indices])
        impurity_right = impurity_functions[self.criterion](y[right_indices])

        weighted_impurity_children = (len(left_indices) / n) * impurity_left + (len(right_indices) / n) * impurity_right
        gain = impurity_before_split - weighted_impurity_children
        return gain
    
    def _split(self, feature_column, threshold_value):
        """Split the data based on the threshold.

        Args:
            feature_column (np.ndarray): Feature values for the selected feature.
            threshold_value (float or str): Threshold value for the split.

        Returns:
            tuple: Indices of the left and right splits.
        """
        if pd.api.types.is_string_dtype(feature_column) and self.one_vs_rest:
            left_indices = np.argwhere(feature_column == threshold_value).flatten()
            right_indices = np.argwhere(feature_column != threshold_value).flatten()
        else:
            left_indices = np.argwhere(feature_column <= threshold_value).flatten()
            right_indices = np.argwhere(feature_column > threshold_value).flatten()

        return left_indices, right_indices
    
    def predict(self, X):
        """Predict labels for the input data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
        np.ndarray: Predicted class labels for each instance.
        """
        print("Predicting labels...")
        predictions = np.array([self._traverse_tree(x) for x in X])
        print("Prediction completed.")
        return predictions

    def _traverse_tree(self, x):
        """Traverse the tree to predict a label for a single instance.

        Args:
            x (np.ndarray): Feature values for a single instance.

        Returns:
            int: Predicted class label for the instance.
        """
        node = self.root
        
        while not node.is_leaf():
            node = node.left_child if x[node.feature_index] <= node.threshold_value else node.right_child

        return node.leaf_value