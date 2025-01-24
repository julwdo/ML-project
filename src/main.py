import pandas as pd
import numpy as np
from src.UDFs import DecisionTreeClassifier, k_fold_nested_cv, train_test_partition
from src.UDFs import accuracy_metric, precision_metric, recall_metric, f1_metric, confusion_matrix

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
model_parameters, test_errors = k_fold_nested_cv(
    X, y, DecisionTreeClassifier, parameter_grid, random_state=42, n_iterations=50
)
mean_test_error = np.mean(test_errors)
min_test_error = np.min(test_errors)
best_model_parameters = model_parameters[np.argmin(test_errors)]

print(f"\nMean Test Error: {mean_test_error:.4f}")
print(f"Minimum Test Error: {min_test_error:.4f}")
print("Best Model Parameters (corresponding to Minimum Test Error):")
for param, value in best_model_parameters.items():
    print(f"  {param}: {value}")
    
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_partition(X, y, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train the final model using the best parameters obtained from nested cross-validation
final_model = DecisionTreeClassifier(**best_model_parameters)
final_model.fit(X_train, y_train)

print("\n### Final Model Trained with Best Parameters ###")

# Make predictions on the test set
y_pred = final_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_metric(y_test, y_pred)
precision = precision_metric(y_test, y_pred)
recall = recall_metric(y_test, y_pred)
f1 = f1_metric(y_test, y_pred)
tp, tn, fp, fn = confusion_matrix(y_test, y_pred)

# Print the evaluation results
print("\n#### Final Model Evaluation ####")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Subsection: Nested Cross-Validation for Model Evaluation (No Missing Values)
print("\n#### Running Model (No Missing Values) with Nested Cross-Validation ####")

# Columns to drop (more than 40% missing values based on the data provided)
columns_to_drop = ['veil-color', 'spore-print-color', 'stem-root', 'stem-surface', 'gill-spacing']

# Drop columns with more than 40% missing values
mushrooms = mushrooms.drop(columns=columns_to_drop)

print(f"Dropped columns: {columns_to_drop}")

# Drop rows with any remaining missing values
rows_before = mushrooms.shape[0]
mushrooms = mushrooms.dropna()
rows_after = mushrooms.shape[0]

print(f"Dropped {rows_before - rows_after} rows with missing values.")
print(f"Remaining rows: {rows_after}, Remaining columns: {mushrooms.shape[1]}")

# Check for missing values
if mushrooms.isnull().any().any():
    print("\nMissing values found in the dataset.")
    na_summary = mushrooms.isnull().sum().loc[lambda x: x > 0].to_frame(name='Missing Count')
    na_summary['Missing Percentage'] = (na_summary['Missing Count'] / n_rows) * 100
    print("\nSummary of missing values:")
    print(na_summary)
else:
    print("\nNo missing values found in the dataset.")
    
# Count edible and poisonous mushrooms
edible_count = (mushrooms['class'] == 0).sum()
poisonous_count = mushrooms.shape[0] - edible_count
print(f"\nNumber of edible mushrooms: {edible_count}")
print(f"Number of poisonous mushrooms: {poisonous_count}")

# Separate features (X) and target (y)
X = mushrooms.drop('class', axis=1).values
y = mushrooms['class'].values

# Perform nested cross-validation
model_parameters, test_errors = k_fold_nested_cv(
    X, y, DecisionTreeClassifier, parameter_grid, random_state=42, n_iterations=50
)
mean_test_error = np.mean(test_errors)
min_test_error = np.min(test_errors)
best_model_parameters = model_parameters[np.argmin(test_errors)]

print(f"\nMean Test Error: {mean_test_error:.4f}")
print(f"Minimum Test Error: {min_test_error:.4f}")
print("Best Model Parameters (corresponding to Minimum Test Error):")
for param, value in best_model_parameters.items():
    print(f"  {param}: {value}")
    
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_partition(X, y, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train the final model using the best parameters obtained from nested cross-validation
final_model = DecisionTreeClassifier(**best_model_parameters)
final_model.fit(X_train, y_train)

print("\n### Final Model Trained with Best Parameters ###")

# Make predictions on the test set
y_pred = final_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_metric(y_test, y_pred)
precision = precision_metric(y_test, y_pred)
recall = recall_metric(y_test, y_pred)
f1 = f1_metric(y_test, y_pred)
tp, tn, fp, fn = confusion_matrix(y_test, y_pred)

# Print the evaluation results
print("\n#### Final Model Evaluation ####")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")