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

# Count edible and poisonous mushrooms
edible_count = (mushrooms['class'] == 'e').sum()
poisonous_count = n_rows - edible_count
print(f"\nNumber of edible mushrooms: {edible_count}")
print(f"Number of poisonous mushrooms: {poisonous_count}")

# Check and handle duplicate rows
duplicates = mushrooms.duplicated().sum()
if duplicates > 0:
    print(f"\nWarning: The dataset contains {duplicates} duplicate rows.")
    mushrooms = mushrooms.drop_duplicates()
    print(f"{duplicates} duplicate rows have been dropped. The dataset now has {mushrooms.shape[0]} rows.")
else:
    print("\nNo duplicate rows found in the dataset.")

# Check for missing values
if mushrooms.isnull().any().any():
    print("\nMissing values found in the dataset.")
    na_summary = mushrooms.isnull().sum().loc[lambda x: x > 0].to_frame(name='Missing Count')
    na_summary['Missing Percentage'] = (na_summary['Missing Count'] / n_rows) * 100
    print("\nSummary of missing values:")
    print(na_summary)
else:
    print("\nNo missing values found in the dataset.")

# Inspect unique values in each column
print("\nUnique values in each column:")
unique_values = {col: mushrooms[col].unique() for col in mushrooms.columns}
for col, values in unique_values.items():
    print(f"- {col}: {values}")

# Identify variables where 'f' is a possible value
variables_with_f = [col for col in mushrooms.columns if 'f' in mushrooms[col].unique()]
print("\nVariables where 'f' is a possible value:", variables_with_f)

# Explore gill-related variables
gill_columns = [col for col in mushrooms.columns if col.startswith('gill')]
f_gill = mushrooms[mushrooms['gill-attachment'] == 'f'][gill_columns]
print("\nGill-related variables when 'gill-attachment' is 'f':")
print(f_gill.drop_duplicates())

# Explore stem-related variables
stem_columns = [col for col in mushrooms.columns if col.startswith('stem')]
f_stem = mushrooms[mushrooms['stem-root'] == 'f'][stem_columns]
print("\nStem-related variables when 'stem-root' is 'f':")
print(f_stem.drop_duplicates())

# Inspect numerical variables
numerical_description = mushrooms.describe()
print("\nSummary of numerical variables:")
print(numerical_description)

# Investigate zero values in 'stem-height' and 'stem-width'
zero_height = mushrooms[mushrooms['stem-height'] == 0][stem_columns]
zero_width = mushrooms[mushrooms['stem-width'] == 0][stem_columns]

print(f"\nRows with 'stem-height' = 0:\n{zero_height.drop_duplicates()}")
print(f"\nRows with 'stem-width' = 0:\n{zero_width.drop_duplicates()}")

# Analyze rows where 'veil-type' is missing
veil_type_missing = mushrooms[mushrooms['veil-type'].isna()]
print("\nClasses of mushrooms where 'veil-type' is missing:")
print(veil_type_missing['class'].unique())

# Drop the 'veil-type' column
mushrooms = mushrooms.drop(columns=['veil-type'])
print("\n'veil-type' column has been dropped from the dataset.")

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

# Subsection: Nested Cross-Validation for Model Evaluation
print("\n#### Running Model with Nested Cross-Validation ####")

# Define the parameter grid for hyperparameter tuning
parameter_grid = {
    'min_samples_split': [2, 5, 10, 20],
    'max_depth': [5, 10, 15, 20, None],
    'n_features': ["log2", "sqrt", None],
    'criterion': ["gini", "scaled_entropy", "square_root"],
    'min_information_gain': [0.0, 0.01, 0.05, 0.1],
    'n_quantiles': [5, 10, 20],
    'isolate_one': [True, False]
}

# Perform nested cross-validation
best_parameters_per_fold, metrics_per_fold = k_fold_nested_cv(
    X, y, DecisionTreeClassifier, parameter_grid, random_state=42, n_iterations=50
)

# Calculate mean of all metrics across folds
mean_metrics = {
    'test_error': np.mean(metrics_per_fold["test_errors"]),
    'precision': np.mean(metrics_per_fold["precisions"]),
    'recall': np.mean(metrics_per_fold["recalls"]),
    'f1_score': np.mean(metrics_per_fold["f1_scores"]),
}

print(f"\nMean Test Error: {mean_metrics['test_error']:.4f}")
print(f"Mean Accuracy: {1 - mean_metrics['test_error']:.4f}")
print(f"Mean Precision: {mean_metrics['precision']:.4f}")
print(f"Mean Recall: {mean_metrics['recall']:.4f}")
print(f"Mean F1 Score: {mean_metrics['f1_score']:.4f}")

# Find the best model (lowest test error)
best_model_index = np.argmin(metrics_per_fold["test_errors"])
best_model_parameters = best_parameters_per_fold[best_model_index]
best_model_metrics = {
    'test_error': metrics_per_fold["test_errors"][best_model_index],
    'accuracy': 1 - metrics_per_fold["test_errors"][best_model_index],
    'precision': metrics_per_fold["precisions"][best_model_index],
    'recall': metrics_per_fold["recalls"][best_model_index],
    'f1_score': metrics_per_fold["f1_scores"][best_model_index],
}

print("\nBest Model Parameters (corresponding to Minimum Test Error):")
for parameter, value in best_model_parameters.items():
    print(f"  {parameter}: {value}")

print(f"\nMetrics for Best Model:")
for metric, value in best_model_metrics.items():
    print(f"  {metric.capitalize()}: {value:.4f}")

# Subsection: Nested Cross-Validation (No Missing Values)
print("\n#### Running Model with Nested Cross-Validation (No Missing Values) ####")

# Drop columns with more than 40% missing values
columns_to_drop = ['veil-color', 'spore-print-color', 'stem-root', 'stem-surface', 'gill-spacing']
mushrooms.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns: {columns_to_drop}")

# Drop rows with any remaining missing values
rows_before = mushrooms.shape[0]
mushrooms.dropna(inplace=True)
rows_after = mushrooms.shape[0]
print(f"Dropped {rows_before - rows_after} rows with missing values.")
print(f"Remaining rows: {rows_after}, Remaining columns: {mushrooms.shape[1]}")

# Check for missing values again
if mushrooms.isnull().any().any():
    print("\nMissing values found in the dataset.")
else:
    print("\nNo missing values found in the dataset.")

# Separate features (X) and target (y) again
X = mushrooms.drop('class', axis=1).values
y = mushrooms['class'].values

# Perform nested cross-validation on cleaned dataset
best_parameters_per_fold, metrics_per_fold = k_fold_nested_cv(
    X, y, DecisionTreeClassifier, parameter_grid, random_state=42, n_iterations=50
)

# Calculate mean of all metrics across folds for cleaned dataset
mean_metrics = {
    'test_error': np.mean(metrics_per_fold["test_errors"]),
    'precision': np.mean(metrics_per_fold["precisions"]),
    'recall': np.mean(metrics_per_fold["recalls"]),
    'f1_score': np.mean(metrics_per_fold["f1_scores"]),
}

print(f"\nMean Test Error: {mean_metrics['test_error']:.4f}")
print(f"Mean Accuracy: {1 - mean_metrics['test_error']:.4f}")
print(f"Mean Precision: {mean_metrics['precision']:.4f}")
print(f"Mean Recall: {mean_metrics['recall']:.4f}")
print(f"Mean F1 Score: {mean_metrics['f1_score']:.4f}")

# Find the best model (lowest test error) for cleaned dataset
best_model_index = np.argmin(metrics_per_fold["test_errors"])
best_model_parameters = best_parameters_per_fold[best_model_index]
best_model_metrics = {
    'test_error': metrics_per_fold["test_errors"][best_model_index],
    'accuracy': 1 - metrics_per_fold["test_errors"][best_model_index],
    'precision': metrics_per_fold["precisions"][best_model_index],
    'recall': metrics_per_fold["recalls"][best_model_index],
    'f1_score': metrics_per_fold["f1_scores"][best_model_index],
}

print("\nBest Model Parameters (corresponding to Minimum Test Error):")
for parameter, value in best_model_parameters.items():
    print(f"  {parameter}: {value}")

print(f"\nMetrics for Best Model:")
for metric, value in best_model_metrics.items():
    print(f"  {metric.capitalize()}: {value:.4f}")