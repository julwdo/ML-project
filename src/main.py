import pandas as pd
from src.UDFs import DecisionTreeClassifier
from src.UDFs import train_test_partition, k_fold_cv_estimate, hyperparameter_tuning, k_fold_nested_cv
from src.UDFs import accuracy_metric

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

# Check for duplicate rows
duplicates = mushrooms.duplicated().sum()
if duplicates > 0:
    print(f"\nWarning: The dataset contains {duplicates} duplicate rows.")
    mushrooms = mushrooms.drop_duplicates()
    print(f"{duplicates} duplicate rows have been dropped. The dataset now has {mushrooms.shape[0]} rows.")
else:
    print("\nNo duplicate rows found in the dataset.")

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
    
# Replace 'f' with NaN in all columns except 'does-bruise-or-bleed' and 'has-ring'
mushrooms_1 = mushrooms.copy()
mushrooms_1.loc[:, ~mushrooms_1.columns.isin(['does-bruise-or-bleed', 'has-ring'])] = \
    mushrooms_1.loc[:, ~mushrooms_1.columns.isin(['does-bruise-or-bleed', 'has-ring'])].replace('f', pd.NA)

# Check for missing values
if not mushrooms_1.isnull().any().any():
    print("\nThere are no missing values in the dataset.")
else:
    print("\nThere are missing values in the dataset.")
    # Summarize missing values
    na_counts = mushrooms_1.isnull().sum()
    na_percentages = (na_counts / n_rows) * 100
    na_summary = pd.DataFrame({
        'Missing Values Count': na_counts,
        'Missing Values Percentage': na_percentages
    }).query('`Missing Values Count` > 0')  # Filter for columns with missing values
    print("\nSummary of missing values:")
    print(na_summary)

# Count the number of edible and poisonous mushrooms
edible_count = (mushrooms['class'] == 'e').sum()
poisonous_count = n_rows - edible_count
print(f"\nNumber of edible mushrooms: {edible_count}")
print(f"Number of poisonous mushrooms: {poisonous_count}")

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

# Subsection: Using Training/Test Split and Arbitrary Hyperparameters
print("\n#### Running Model with Training/Test Split and Arbitrary Hyperparameters ####")

# Partition the dataset into training and testing sets (default: 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_partition(X, y, random_state=42)

# Instantiate the Decision Tree Classifier with arbitrary hyperparameters
tree = DecisionTreeClassifier(n_features="log2", n_quantiles=10)

# Train the classifier on the training data
tree.fit(X_train, y_train)

# Predict on training data to compute training error
y_train_predicted = tree.predict(X_train)
training_error = 1 - accuracy_metric(y_train, y_train_predicted)
print(f"\nTraining Results:")
print(f"Training error: {training_error:.2f}")

# Predict on test data to compute test accuracy and error
y_test_predicted = tree.predict(X_test)
accuracy = accuracy_metric(y_test, y_test_predicted)
test_error = 1 - accuracy
print(f"\nTesting Results:")
print(f"Accuracy (test set): {accuracy:.2f}")
print(f"Test error: {test_error:.2f}")

# Subsection: Using Cross-Validation and Arbitrary Hyperparameters
print("\n#### Running Model with Cross-Validation and Arbitrary Hyperparameters ####")

# Set arbitrary hyperparameters
parameters = {
    'n_features': "log2",
    'n_quantiles': 10
}

# Perform k-fold cross-validation
mean_test_error = k_fold_cv_estimate(X, y, DecisionTreeClassifier, parameters, random_state=42)
print(f"Mean test error: {mean_test_error:.2f}")

# Subsection: Using Training/Test Split and Tuned Hyperparameters
print("\n#### Running Model with Training/Test Split and Tuned Hyperparameters ####")

# Split the data into training and testing sets again for hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_partition(X, y, random_state=42)

# Define the parameter grid for hyperparameter tuning
parameter_grid = {
    'min_samples_split': [2],
    'max_depth': [15, 20],
    'n_features': ["log2"],
    'criterion': ["gini", "scaled_entropy"],
    'min_information_gain': [0.0],
    'n_quantiles': [5, 10],
    'isolate_one': [False]
}

# Perform hyperparameter tuning
best_parameters, best_mean_test_error = hyperparameter_tuning(X_train, y_train, DecisionTreeClassifier, parameter_grid, random_state=42)
print("Best parameters:", best_parameters)
print("Best mean test error:", best_mean_test_error)

# Subsection: Using Nested Cross-Validation
print("\n#### Running Model with Nested Cross-Validation ####")

# Perform nested cross-validation with the same parameter grid
k_fold_nested_cv(X, y, DecisionTreeClassifier, parameter_grid, random_state=42)