import pandas as pd
import numpy as np
from src.UDFs import DecisionTreeClassifier, k_fold_nested_cv

# Section: Load and Explore the Dataset
print("### Mushroom Dataset Exploration ###")

# Load the dataset
mushrooms = pd.read_csv('data/secondary_data.csv', delimiter=';')

# Basic dataset information
n_rows, n_columns = mushrooms.shape
print(f"The dataset contains {n_rows} rows and {n_columns} columns.")
print("\nFirst few rows of the dataset:")
print(mushrooms.head())
print("\nColumn data types:")
print(mushrooms.dtypes)

# Count the number of edible and poisonous mushrooms
edible_count = (mushrooms['class'] == 'e').sum()
poisonous_count = n_rows - edible_count
print(f"\nNumber of edible mushrooms: {edible_count}")
print(f"Number of poisonous mushrooms: {poisonous_count}")

# Check for duplicate rows
duplicates = mushrooms.duplicated().sum()
if duplicates > 0:
    print(f"\nWarning: The dataset contains {duplicates} duplicate rows.")
    mushrooms = mushrooms.drop_duplicates()
    print(f"{duplicates} duplicate rows have been dropped. The dataset now has {mushrooms.shape[0]} rows.")
else:
    print("\nNo duplicate rows found in the dataset.")
    
# Count the number of edible and poisonous mushrooms
edible_count = (mushrooms['class'] == 'e').sum()
poisonous_count = mushrooms.shape[0] - edible_count
print(f"\nNumber of edible mushrooms: {edible_count}")
print(f"Number of poisonous mushrooms: {poisonous_count}")

# Check for missing values
if not mushrooms.isnull().any().any():
    print("\nThere are no missing values in the dataset.")
else:
    print("\nThere are missing values in the dataset.")
    # Summarize missing values
    na_counts = mushrooms.isnull().sum()
    na_percentages = (na_counts / n_rows) * 100
    na_summary = pd.DataFrame({
        'Missing Values Count': na_counts,
        'Missing Values Percentage': na_percentages
    }).query('`Missing Values Count` > 0')  # Filter for columns with missing values
    print("\nSummary of missing values:")
    print(na_summary)

# Inspect unique values in each column
print("\nUnique values in each column:")
for col in mushrooms.columns:
    print(f"- {col}: {mushrooms[col].unique()}")
    
# Inspect numerical variable values
mushrooms.describe()
(mushrooms['stem-height'] == 0).sum()
(mushrooms['stem-width'] == 0).sum()

mushrooms[mushrooms['stem-height'] == 0]['stem-width'].unique() # Only 0
mushrooms[mushrooms['stem-width'] == 0]['stem-height'].unique() # Only 0

mushrooms[mushrooms['stem-height'] == 0]['stem-root'].unique() # Only f
mushrooms[mushrooms['stem-width'] == 0]['stem-root'].unique() # Only f

mushrooms[mushrooms['stem-height'] == 0]['stem-surface'].unique() # Only f
mushrooms[mushrooms['stem-width'] == 0]['stem-surface'].unique() # Only f

mushrooms[mushrooms['stem-height'] == 0]['stem-color'].unique() # Only f
mushrooms[mushrooms['stem-width'] == 0]['stem-color'].unique() # Only f
    
# Replace 'f' with NaN in all columns except 'does-bruise-or-bleed' and 'has-ring'
mushrooms.loc[:, ~mushrooms.columns.isin(['does-bruise-or-bleed', 'has-ring'])] = \
    mushrooms.loc[:, ~mushrooms.columns.isin(['does-bruise-or-bleed', 'has-ring'])].replace('f', pd.NA)
    
# Replace 0 in 'stem-height' and 'stem-width' with NaN
mushrooms.loc[:, mushrooms.columns.isin(['stem-height', 'stem-width'])] = \
    mushrooms.loc[:, mushrooms.columns.isin(['stem-height', 'stem-width'])].replace(0., pd.NA)

# Check for missing values
if not mushrooms.isnull().any().any():
    print("\nThere are no missing values in the dataset.")
else:
    print("\nThere are missing values in the dataset.")
    # Summarize missing values
    na_counts = mushrooms.isnull().sum()
    na_percentages = (na_counts / n_rows) * 100
    na_summary = pd.DataFrame({
        'Missing Values Count': na_counts,
        'Missing Values Percentage': na_percentages
    }).query('`Missing Values Count` > 0')  # Filter for columns with missing values
    print("\nSummary of missing values:")
    print(na_summary)

# Section: Preprocessing and Classification
print("\n### Mushroom Classification ###")

# Map 'class' column values to numeric labels: 'e' -> 0 (edible), 'p' -> 1 (poisonous)
if set(mushrooms['class'].unique()) == {'e', 'p'}:
    mushrooms['class'] = mushrooms['class'].map({'e': 0, 'p': 1})
else:
    raise ValueError("Unexpected values in the 'class' column.")

# Separate features (X) and target (y)
X = mushrooms.drop('class', axis=1).values
y = mushrooms['class'].values

# Subsection: Using Nested Cross-Validation
print("\n#### Running Model with Nested Cross-Validation ####")

# Define the parameter grid for hyperparameter tuning
parameter_grid = {
    'min_samples_split': [2],
    'max_depth': [15],
    'n_features': ["log2"],
    'criterion': ["gini", "scaled_entropy"],
    'min_information_gain': [0.0],
    'n_quantiles': [5],
    'isolate_one': [False]
}

# Perform nested cross-validation
model_parameters, test_errors = k_fold_nested_cv(X, y, DecisionTreeClassifier, parameter_grid, random_state=42)

mean_test_error = np.mean(test_errors)
min_test_error = np.min(test_errors)
best_model_params = model_parameters[np.argmin(test_errors)]

print(f"Mean Test Error: {mean_test_error:.4f}")
print(f"Minimum Test Error: {min_test_error:.4f}")
print(f"Best Model Parameters (corresponding to Minimum Test Error): {best_model_params}")

# Dropping columns with more than 45% of missing values
mushrooms.drop(['gill-spacing', 'gill-spacing', 'stem-surface', 'veil-type', 'veil-color', 'ring-type',
                'spore-print-color'], axis=1)

# Dropping subsequent rows with missing values
mushrooms.dropna()

# Perform nested cross-validation with the same parameter grid
