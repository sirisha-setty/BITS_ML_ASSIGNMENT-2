# Import os library
import os

# Define the model folder
model_folder = "model"

# List all files in the model folder
files = os.listdir(model_folder)

# Display saved files
print("========== SAVED MODEL FILES ==========")

for file in files:
    print(file)

# Required files
required_files = [
    "logistic_regression_model.joblib",
    "decision_tree_model.joblib",
    "knn_model.joblib",
    "naive_bayes_model.joblib",
    "svm_model.joblib",
    "scaler.joblib",
    "feature_columns.joblib",
    "label_mapping.joblib"
]

# Verify each file
print("\n========== FILE VERIFICATION ==========")

for file in required_files:

    if os.path.exists(os.path.join(model_folder, file)):
        print(file, "- FOUND")
    else:
        print(file, "- MISSING")

        