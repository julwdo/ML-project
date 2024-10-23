from src.UDFs import DecisionTreeClassifier, compute_accuracy
import numpy as np

X = np.array([[1, 2],
              [1, 3],
              [2, 2],
              [2, 3],
              [3, 1],
              [3, 2]])
y = np.array([0, 0, 1, 1, 0, 1])  # Example binary labels

# Create and fit the decision tree
tree = DecisionTreeClassifier(min_samples_split=5, criterion="entropy")
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
print("Predictions:", predictions)

# Calculate accuracy
accuracy = compute_accuracy(y, predictions)
print(f"Accuracy: {accuracy:.2f}")