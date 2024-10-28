import numpy as np
import pandas as pd

def compute_classification_error(y):
    """Calculate the classification error of labels."""
    probabilities = np.bincount(y) / len(y)
    return 1 - np.max(probabilities)

def compute_entropy(y):
    """Calculate the entropy of labels."""
    probabilities = np.bincount(y) / len(y)
    return -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))

def compute_gini(y):
    """Calculate the Gini impurity of labels."""
    probabilities = np.bincount(y) / len(y)
    return 1.0 - np.sum(probabilities ** 2)

def train_test_partition(X, y, test_size=0.2, random_state=None):
    """Partition data into training and testing sets."""
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_index = int(len(indices) * (1 - test_size))
    
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def k_fold_partition(X, k=5, random_state=None):
    """Split data into k folds for cross-validation."""
    
    if random_state is not None:
        np.random.seed(random_state)
        
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    fold_size = X.shape[0] // k
    fold_indices = []
    
    for i in range(k):
        val_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        
        fold_indices.append((train_indices, val_indices))
    
    return fold_indices

def zero_one_loss(y, y_predicted):
    """Compute zero-one loss."""
    return np.sum(y != y_predicted)

def accuracy_metric(y, y_predicted):
    """Compute accuracy."""
    return np.sum(y == y_predicted) / len(y)

class TreeNode:
    def __init__(self, feature_index=None, threshold_value=None, left_child=None, right_child=None, left_ratio=None, leaf_value=None):
        """Initialize a tree node."""
        self.feature_index = feature_index
        self.threshold_value = threshold_value
        self.left_child = left_child
        self.right_child = right_child
        self.left_ratio = left_ratio
        self.leaf_value = leaf_value

    def is_leaf(self):
        """Check if the node is a leaf."""
        return self.leaf_value is not None
    
class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=None, n_features=None, 
                 criterion="entropy", min_information_gain=0.0, n_quantiles=None, 
                 isolate_one=False):
        """Initialize the Decision Tree Classifier."""
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.min_information_gain = min_information_gain
        self.n_quantiles = n_quantiles
        self.isolate_one = isolate_one
        self.root = None
        self.depth = 0  # To track the final depth of the tree
        
    def fit(self, X, y):
        """Fit the model to the training data."""
        print("Fitting the model...")
        
        n_features_functions = {
            "sqrt": np.sqrt,
            "log2": np.log2
        }
        
        if isinstance(self.n_features, int):
            self.n_features = min(self.n_features or X.shape[1], X.shape[1])
        elif isinstance(self.n_features, float):
            self.n_features = max(1, int(self.n_features * X.shape[1]))
        else:
            self.n_features = max(1, int(n_features_functions[self.n_features](X.shape[1])))
        
        print(f"Considering {self.n_features} features at each split.")
        
        self.root = self._build_tree(X, y)
        print(f"Model fitting completed. Final depth: {self.depth}")

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        self.depth = max(self.depth, depth)  # Update the final depth of the tree      
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
        
        # Keep track of the ratio of samples going to the left child
        left_ratio = len(left_indices) / n_samples
        
        # Print the feature and threshold used to split, and print the values going to the left and right
        print(f"Best feature index: {best_feature_index}")
        print(f"Best threshold value: {best_threshold_value}")

        # Create child nodes
        left_child = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)
        
        return TreeNode(best_feature_index, best_threshold_value, left_child, right_child, left_ratio)
    
    def _get_most_common_label(self, y):
        """Find the most common label in the labels array."""
        unique_labels, counts = np.unique(y, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        return most_common_label
    
    def _find_best_split(self, X, y, feature_indices):
        """Determine the best feature and threshold for splitting."""
        best_gain = self.min_information_gain
        best_feature_index, best_threshold_value = None, None
        
        for feature_index in feature_indices:
            feature_column = X[:, feature_index]
            
            not_missing_indices = np.argwhere(~pd.isnull(feature_column)).flatten()
            
            if pd.api.types.is_string_dtype(feature_column[not_missing_indices]): # Ignore missing values when computing thresholds
                unique_thresholds = np.unique(feature_column[not_missing_indices])
            else:
                unique_values = np.unique(feature_column[not_missing_indices])
                
                if self.n_quantiles is None:
                    unique_thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                else:
                    unique_thresholds = np.quantile(unique_values, np.linspace(0, 1, self.n_quantiles + 1)[1:-1])
            
            # Print the thresholds being considered for the current feature
            print(f"Considering thresholds for feature index {feature_index}: {unique_thresholds}")
            
            for threshold_value in unique_thresholds:
                gain = self._calculate_information_gain(y, feature_column, threshold_value)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold_value = threshold_value

        return best_feature_index, best_threshold_value

    def _calculate_information_gain(self, y, feature_column, threshold_value):
        """Calculate the information gain from a split based on the selected criterion."""
        impurity_functions = {
            "entropy": compute_entropy,
            "gini": compute_gini,
            "classification_error": compute_classification_error
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
        """Split the data based on the threshold."""
        missing_indices = np.argwhere(pd.isnull(feature_column)).flatten()
        not_missing_indices = np.argwhere(~pd.isnull(feature_column)).flatten()
        
        # Split non-missing values
        if pd.api.types.is_string_dtype(feature_column) and self.isolate_one:
            left_indices = not_missing_indices[feature_column[not_missing_indices] == threshold_value]
        else:
            left_indices = not_missing_indices[feature_column[not_missing_indices] <= threshold_value]
        
        right_indices = np.setdiff1d(not_missing_indices, left_indices)
            
        # Randomly distribute missing values proportionally based on the sizes of left and right splits
        left_ratio = len(left_indices) / len(not_missing_indices)
        
        left_missing = np.random.choice(missing_indices, int(left_ratio * len(missing_indices)), replace=False)
        right_missing = np.setdiff1d(missing_indices, left_missing)
           
        left_indices = np.concatenate([left_indices, left_missing])
        right_indices = np.concatenate([right_indices, right_missing])
        
        return left_indices, right_indices
    
    def predict(self, X):
        """Predict labels for the input data."""
        print("Predicting labels...")
        predictions = np.array([self._traverse_tree(x) for x in X])
        print("Prediction completed.")
        return predictions
    
    def _traverse_tree(self, x):
        """Traverse the tree to predict a label for a single instance."""
        node = self.root
        while not node.is_leaf():
            feature_value = x[node.feature_index]
            
            if pd.isnull(feature_value):
                if np.random.binomial(1, node.left_ratio) == 1:
                    node = node.left_child
                else:
                    node = node.right_child
                
            else:
                node = node.left_child if feature_value <= node.threshold_value else node.right_child
        return node.leaf_value