from src.UDFs import Tree
import numpy as np

X = np.array([[1, 2],
              [1, 3],
              [2, 2],
              [2, 3],
              [3, 1],
              [3, 2]])
y = np.array([0, 0, 1, 1, 0, 1])  # Example binary labels

# Create and fit the decision tree
tree = Tree(min_samples_split=5, max_depth=3, n_features=2)
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
print("Predictions:", predictions)