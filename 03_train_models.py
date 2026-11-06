
# =========================================
# BITS PILANI - ML ASSIGNMENT 2
# STEP 17: LOGISTIC REGRESSION
# =========================================

# STEP 1: Import required libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# STEP 2: Define dataset column names

column_names = [
    "ID", "Diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]


# STEP 3: Load the dataset

data = pd.read_csv(
    "dataset/wdbc.data",
    header=None,
    names=column_names
)


# STEP 4: Separate features and target

X = data.drop(columns=["ID", "Diagnosis"])

y = data["Diagnosis"].map({
    "B": 0,
    "M": 1
})


# STEP 5: Split into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# STEP 6: Feature scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# STEP 7: Create Logistic Regression model

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)


# STEP 8: Train the model

logistic_model.fit(X_train_scaled, y_train)


# STEP 9: Make predictions

y_pred_logistic = logistic_model.predict(X_test_scaled)


# STEP 10: Calculate prediction probabilities

y_prob_logistic = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


# STEP 11: Display results

print("\nActual Diagnosis:")
print(y_test.head(10).values)

print("\nPredicted Diagnosis:")
print(y_pred_logistic[:10])

print("\nMalignant Prediction Probabilities:")
print(y_prob_logistic[:10])

print("\nLogistic Regression training completed successfully!")


# =========================================
# STEP 18: EVALUATE LOGISTIC REGRESSION
# =========================================

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_logistic)

auc = roc_auc_score(y_test, y_prob_logistic)

precision = precision_score(y_test, y_pred_logistic)

recall = recall_score(y_test, y_pred_logistic)

f1 = f1_score(y_test, y_pred_logistic)

mcc = matthews_corrcoef(y_test, y_pred_logistic)


# Display evaluation results
print("\n========== LOGISTIC REGRESSION RESULTS ==========")

print("Accuracy:", round(accuracy, 4))
print("AUC:", round(auc, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("MCC:", round(mcc, 4))


# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))


# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))


# =========================================
# STEP 19: DECISION TREE CLASSIFIER
# =========================================

# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree model
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

# Train the model using training data
decision_tree_model.fit(X_train, y_train)

# Predict diagnosis using test data
y_pred_tree = decision_tree_model.predict(X_test)

# Calculate prediction probabilities
y_prob_tree = decision_tree_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy_tree = accuracy_score(y_test, y_pred_tree)

auc_tree = roc_auc_score(y_test, y_prob_tree)

precision_tree = precision_score(y_test, y_pred_tree)

recall_tree = recall_score(y_test, y_pred_tree)

f1_tree = f1_score(y_test, y_pred_tree)

mcc_tree = matthews_corrcoef(y_test, y_pred_tree)

# Display results
print("\n========== DECISION TREE RESULTS ==========")

print("Accuracy:", round(accuracy_tree, 4))
print("AUC:", round(auc_tree, 4))
print("Precision:", round(precision_tree, 4))
print("Recall:", round(recall_tree, 4))
print("F1 Score:", round(f1_tree, 4))
print("MCC:", round(mcc_tree, 4))

# Display confusion matrix
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

# Display classification report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))


# =========================================
# STEP 21: K-NEAREST NEIGHBORS CLASSIFIER
# =========================================

# Import KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Create KNN model with 5 neighbors
knn_model = KNeighborsClassifier(
    n_neighbors=5
)

# Train the KNN model
knn_model.fit(X_train_scaled, y_train)

# Predict diagnosis
y_pred_knn = knn_model.predict(X_test_scaled)

# Calculate prediction probabilities
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

# Calculate evaluation metrics
accuracy_knn = accuracy_score(y_test, y_pred_knn)

auc_knn = roc_auc_score(y_test, y_prob_knn)

precision_knn = precision_score(y_test, y_pred_knn)

recall_knn = recall_score(y_test, y_pred_knn)

f1_knn = f1_score(y_test, y_pred_knn)

mcc_knn = matthews_corrcoef(y_test, y_pred_knn)

# Display results
print("\n========== KNN RESULTS ==========")

print("Accuracy:", round(accuracy_knn, 4))
print("AUC:", round(auc_knn, 4))
print("Precision:", round(precision_knn, 4))
print("Recall:", round(recall_knn, 4))
print("F1 Score:", round(f1_knn, 4))
print("MCC:", round(mcc_knn, 4))

# Display confusion matrix
print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# Display classification report
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))



# =========================================
# STEP 24: GAUSSIAN NAIVE BAYES
# =========================================

# Import Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Create Gaussian Naive Bayes model
naive_bayes_model = GaussianNB()

# Train the model
naive_bayes_model.fit(X_train_scaled, y_train)

# Predict diagnosis
y_pred_nb = naive_bayes_model.predict(X_test_scaled)

# Calculate prediction probabilities
y_prob_nb = naive_bayes_model.predict_proba(X_test_scaled)[:, 1]

# Calculate evaluation metrics
accuracy_nb = accuracy_score(y_test, y_pred_nb)

auc_nb = roc_auc_score(y_test, y_prob_nb)

precision_nb = precision_score(y_test, y_pred_nb)

recall_nb = recall_score(y_test, y_pred_nb)

f1_nb = f1_score(y_test, y_pred_nb)

mcc_nb = matthews_corrcoef(y_test, y_pred_nb)

# Display results
print("\n========== GAUSSIAN NAIVE BAYES RESULTS ==========")

print("Accuracy:", round(accuracy_nb, 4))
print("AUC:", round(auc_nb, 4))
print("Precision:", round(precision_nb, 4))
print("Recall:", round(recall_nb, 4))
print("F1 Score:", round(f1_nb, 4))
print("MCC:", round(mcc_nb, 4))

# Display confusion matrix
print("\nNaive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# Display classification report
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))


# =========================================
# STEP 26: SUPPORT VECTOR MACHINE (SVM)
# =========================================

# Import Support Vector Classifier
from sklearn.svm import SVC

# Create the SVM model
# probability=True allows us to calculate AUC
svm_model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

# Train the SVM model using scaled training data
svm_model.fit(X_train_scaled, y_train)

# Predict diagnosis for test data
y_pred_svm = svm_model.predict(X_test_scaled)

# Calculate probability of malignant diagnosis (class 1)
y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# Calculate evaluation metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_prob_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
mcc_svm = matthews_corrcoef(y_test, y_pred_svm)

# Display evaluation results
print("\n========== SUPPORT VECTOR MACHINE RESULTS ==========")
print("Accuracy:", round(accuracy_svm, 4))
print("AUC:", round(auc_svm, 4))
print("Precision:", round(precision_svm, 4))
print("Recall:", round(recall_svm, 4))
print("F1 Score:", round(f1_svm, 4))
print("MCC:", round(mcc_svm, 4))

# Display confusion matrix
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# Display classification report
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# =========================================
# STEP 28: COMPARE ALL FIVE MODELS
# =========================================

# Create a comparison table
comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Gaussian Naive Bayes",
        "SVM"
    ],

    "Accuracy": [
        accuracy,
        accuracy_tree,
        accuracy_knn,
        accuracy_nb,
        accuracy_svm
    ],

    "AUC": [
        auc,
        auc_tree,
        auc_knn,
        auc_nb,
        auc_svm
    ],

    "Precision": [
        precision,
        precision_tree,
        precision_knn,
        precision_nb,
        precision_svm
    ],

    "Recall": [
        recall,
        recall_tree,
        recall_knn,
        recall_nb,
        recall_svm
    ],

    "F1 Score": [
        f1,
        f1_tree,
        f1_knn,
        f1_nb,
        f1_svm
    ],

    "MCC": [
        mcc,
        mcc_tree,
        mcc_knn,
        mcc_nb,
        mcc_svm
    ]
})

# Display comparison table
print("\n========== FINAL MODEL COMPARISON ==========")

print(comparison.round(4))

# Save comparison table as CSV
comparison.to_csv("model_comparison.csv", index=False)

print("\nModel comparison saved successfully!")


# =========================================
# STEP 29: SAVE ALL FIVE TRAINED MODELS
# =========================================

# Import joblib
import joblib

# Save Logistic Regression model
joblib.dump(logistic_model, "model/logistic_regression_model.joblib")

# Save Decision Tree model
joblib.dump(decision_tree_model, "model/decision_tree_model.joblib")

# Save KNN model
joblib.dump(knn_model, "model/knn_model.joblib")

# Save Gaussian Naive Bayes model
joblib.dump(naive_bayes_model, "model/naive_bayes_model.joblib")

# Save SVM model
joblib.dump(svm_model, "model/svm_model.joblib")

print("\nAll five models saved successfully!")


# =========================================
# STEP 30: SAVE THE SCALER
# =========================================

# Save the fitted StandardScaler
joblib.dump(scaler, "model/scaler.joblib")

print("\nStandardScaler saved successfully!")

# =========================================
# STEP 31: SAVE FEATURE COLUMNS AND LABEL MAPPING
# =========================================

# Save the feature names and their order
joblib.dump(X.columns.tolist(), "model/feature_columns.joblib")

# Save the diagnosis label mapping
label_mapping = {
    "B": 0,
    "M": 1
}

joblib.dump(label_mapping, "model/label_mapping.joblib")

print("\nFeature columns saved successfully!")
print("Label mapping saved successfully!")

# ============================================================
# STEP 32: SAVE TEST DATA FOR STREAMLIT APPLICATION
# ============================================================

# Create a copy of the test feature data
test_data = X_test.copy()

# Add the actual diagnosis values
test_data["Diagnosis"] = y_test.values

# Save the test data as a CSV file in the project folder
test_data.to_csv("test_data.csv", index=False)

# Display confirmation and dataset dimensions
print("\nTest data saved successfully!")
print("Test data shape:", test_data.shape)



