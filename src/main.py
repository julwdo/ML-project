import pandas as pd
import numpy as np
from src.UDFs import DecisionTreeClassifier, compute_accuracy

mushrooms = pd.read_csv('data/secondary_data.csv', delimiter = ';')

mushrooms['class'] = mushrooms['class'].map({'e': 0, 'p': 1})

mushrooms_1 = mushrooms.dropna(axis=1)

X = mushrooms_1.drop('class', axis=1).values
y = mushrooms_1['class'].values

tree = DecisionTreeClassifier(max_depth=20, n_features=5)
tree.fit(X, y)

predictions = tree.predict(X)
print("Predictions:", predictions)

# Calculate accuracy
accuracy = compute_accuracy(y, predictions)
print(f"Accuracy: {accuracy:.2f}")