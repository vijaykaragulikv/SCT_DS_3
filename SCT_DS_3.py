import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# --- Phase 1: Data Acquisition and Loading ---
# Using the recommended 'bank-full' file.
# IMPORTANT: Adjust the file name/extension ('bank-full.csv' or 'bank-full.xlsx') 
# and separator (sep=';') based on how your file is saved.
try:
    # Assuming CSV file with semicolon separator (common for this dataset)
    # Assuming your file is named 'bank-full.csv' and is in that directory
    df = pd.read_csv('D:\\lab programs\\Internship_tasks\\SCT_DS_3\\datasets\\bank-full.csv', sep=';')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: File not found. Please check the file name and path.")
    exit()
except pd.errors.ParserError:
    print("Error: Could not parse file. Check if the separator (sep=';') is correct.")
    exit()

# --- Phase 2: Data Preprocessing and Feature Engineering ---

# 1. Handle Target Variable ('y')
# Convert 'yes' to 1 (purchased) and 'no' to 0 (did not purchase)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# 2. Separate Features (X) and Target (y)
X = df.drop('y', axis=1)
y = df['y']

# 3. One-Hot Encoding for Categorical Features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Apply One-Hot Encoding only to the feature set X
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("\n--- Encoded Data Shape ---")
print(f"Number of Features after encoding: {X_encoded.shape[1]}")

# 4. Split the Data into Training and Testing Sets
# 'stratify=y' ensures the class ratio (purchased/not purchased) is maintained in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

print("\n--- Data Split ---")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# --- Phase 3 & 4: Model Building, Hyperparameter Tuning, and Evaluation ---

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'max_depth': [3, 5, 7, 10, None], # Controls complexity and helps prevent overfitting
    'min_samples_leaf': [5, 10, 20],   # Ensures splits are based on enough data
    'criterion': ['gini', 'entropy']   # Different ways to measure split quality
}

# Setup Grid Search with Cross-Validation (cv=5)
# FIX APPLIED: n_jobs=1 forces sequential execution, avoiding the '_posixsubprocess' error.
grid_search = GridSearchCV(
    estimator=dt_classifier,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc', # ROC AUC is better for imbalanced data than simple accuracy
    n_jobs=1,          # <-- CRITICAL CHANGE: Set to 1
    verbose=2          # Increased verbosity to see progress
)

# Run the Grid Search on the training data
print("\n--- Starting Grid Search for Hyperparameter Tuning (n_jobs=1) ---")
grid_search.fit(X_train, y_train)

# Get the best estimator (the best tuned model)
best_dt_model = grid_search.best_estimator_

print("\n--- Best Model Parameters ---")
print(grid_search.best_params_)

# 5. Final Prediction and Evaluation
y_pred = best_dt_model.predict(X_test)

print("\n--- Model Performance on Test Set (Best Model) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Visualize the Decision Tree (Optional but Recommended)
plt.figure(figsize=(20, 10))
# Plot the tree, limiting depth for readability
plot_tree(
    best_dt_model,
    feature_names=X_encoded.columns.tolist(),
    class_names=['No Purchase (0)', 'Purchase (1)'],
    filled=True,
    rounded=True,
    max_depth=3 # Visualize only the top 3 levels for clarity
)
plt.title(f"Optimized Decision Tree (Top 3 Levels) - Best Max Depth: {best_dt_model.get_params()['max_depth']}")
plt.show()