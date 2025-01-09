import pandas as pd
import numpy as np
from UDFs import DecisionTreeClassifier, compute_accuracy

mushrooms = pd.read_csv('data/secondary_data.csv', delimiter = ';')

mushrooms['class'] = mushrooms['class'].map({'e': 0, 'p': 1})

#mushrooms_1 = mushrooms.dropna(axis=1)

X = mushrooms.drop('class', axis=1).values
y = mushrooms['class'].values

X

y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])

def _gini_impurity(self, y):
    """Calculate the Gini impurity of labels."""
    probabilities = np.bincount(y) / len(y)
    print("Probabilities:", probabilities)  # Print the probabilities to check their values
    return 2 * probabilities[0] * (1 - probabilities[0])

