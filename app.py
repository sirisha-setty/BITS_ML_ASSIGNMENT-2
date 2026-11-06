
# ============================================================
# BREAST CANCER CLASSIFICATION
# BITS PILANI WILP M.TECH AI & ML - ASSIGNMENT 2
# ============================================================

# STEP 1: IMPORT REQUIRED LIBRARIES

import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

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


# STEP 2: CONFIGURE STREAMLIT PAGE

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide"
)


# STEP 3: DEFINE PROJECT AND MODEL PATHS

# Get the folder in which app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Define the model folder
MODEL_FOLDER = BASE_DIR / "model"


# STEP 4: LOAD SAVED MACHINE LEARNING MODELS

# Dictionary containing model names and their filenames
model_files = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "K-Nearest Neighbors": "knn_model.joblib",
    "Gaussian Naive Bayes": "naive_bayes_model.joblib",
    "Support Vector Machine": "svm_model.joblib"
}

# Create an empty dictionary to store models
models = {}

# Load all five trained models
try:

    for model_name, file_name in model_files.items():

        file_path = MODEL_FOLDER / file_name

        models[model_name] = joblib.load(file_path)

    # Load the saved StandardScaler
    scaler = joblib.load(MODEL_FOLDER / "scaler.joblib")

    # Load the feature column names
    feature_columns = joblib.load(
        MODEL_FOLDER / "feature_columns.joblib"
    )

    # Load the diagnosis label mapping
    label_mapping = joblib.load(
        MODEL_FOLDER / "label_mapping.joblib"
    )

except Exception as e:

    st.error(f"Unable to load saved models or preprocessing files: {e}")
    st.stop()


# STEP 5: DISPLAY APPLICATION TITLE AND INTRODUCTION

st.title("Breast Cancer Classification")

st.write(
    """
    This application uses five Machine Learning classification
    algorithms to classify breast tumours as Benign or Malignant.

    Upload a test dataset, select a trained model, and evaluate
    its performance using standard classification metrics.
    """
)

st.subheader("Available Classification Models")

st.write(
    """
    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors (KNN)
    4. Gaussian Naive Bayes
    5. Support Vector Machine (SVM)
    """
)

st.info(
    "Upload a CSV file containing the 30 input features "
    "and the actual Diagnosis column."
)


# STEP 6: UPLOAD TEST DATASET

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a test dataset",
    type=["csv"]
)


# STEP 7: SELECT MACHINE LEARNING MODEL

selected_model = st.selectbox(
    "Select a Machine Learning Model",
    list(models.keys())
)


# STEP 8: PROCESS UPLOADED DATASET

if uploaded_file is not None:

    try:

        # Read uploaded CSV file
        test_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Information")

        st.write("Dataset shape:", test_data.shape)

        # Check whether dataset is empty
        if test_data.empty:

            st.error("The uploaded dataset is empty.")

        # Check whether Diagnosis column exists
        elif "Diagnosis" not in test_data.columns:

            st.error(
                "The uploaded CSV must contain a Diagnosis column."
            )

        else:

            # Separate input features and actual diagnosis
            X_test = test_data.drop(columns=["Diagnosis"])

            y_test = test_data["Diagnosis"]

            # Check whether required features are available
            missing_features = [
                col for col in feature_columns
                if col not in X_test.columns
            ]

            if missing_features:

                st.error(
                    f"Missing required features: {missing_features}"
                )

            else:

                # Arrange features in the correct training order
                X_test = X_test[feature_columns]

                # Convert feature columns into numeric values
                X_test = X_test.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                # Convert diagnosis labels using saved mapping
                y_test = y_test.replace(label_mapping)

                # Convert diagnosis values into numeric values
                y_test = pd.to_numeric(
                    y_test,
                    errors="coerce"
                )

                # Check for missing or invalid values
                if X_test.isnull().any().any() or y_test.isnull().any():

                    st.error(
                        "The dataset contains missing or invalid values."
                    )

                # Check whether diagnosis contains only 0 and 1
                elif not set(y_test.unique()).issubset({0, 1}):

                    st.error(
                        "Diagnosis must contain only 0 (Benign) "
                        "or 1 (Malignant)."
                    )

                # AUC requires both classes to be present
                elif len(y_test.unique()) < 2:

                    st.error(
                        "The test dataset must contain both "
                        "Benign and Malignant cases to calculate AUC."
                    )

                else:

                    st.success("Dataset validation successful!")

                    # STEP 9: GENERATE MODEL PREDICTIONS

                    st.subheader("Prediction Results")

                    # Retrieve the selected saved model
                    model = models[selected_model]

                    # Apply scaling where required
                    if selected_model == "Decision Tree":

                        X_input = X_test

                    else:

                        X_input = scaler.transform(X_test)

                    # Generate predicted diagnosis
                    y_pred = model.predict(X_input)

                    # Generate malignant probabilities
                    y_prob = model.predict_proba(X_input)[:, 1]

                    # Display actual and predicted results
                    results = pd.DataFrame({
                        "Actual Diagnosis": y_test.values,
                        "Predicted Diagnosis": y_pred,
                        "Malignant Probability": y_prob
                    })

                    st.dataframe(results)

                    st.write(
                        "Total Predictions:",
                        len(y_pred)
                    )


                    # STEP 10: CALCULATE EVALUATION METRICS

                    st.subheader("Model Performance Metrics")

                    accuracy = accuracy_score(y_test, y_pred)

                    auc = roc_auc_score(y_test, y_prob)

                    precision = precision_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )

                    recall = recall_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )

                    mcc = matthews_corrcoef(y_test, y_pred)

                    # Display metrics in two rows
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Accuracy", f"{accuracy:.4f}")

                    col2.metric("AUC", f"{auc:.4f}")

                    col3.metric("Precision", f"{precision:.4f}")

                    col4, col5, col6 = st.columns(3)

                    col4.metric("Recall", f"{recall:.4f}")

                    col5.metric("F1 Score", f"{f1:.4f}")

                    col6.metric("MCC", f"{mcc:.4f}")


                    # STEP 11: DISPLAY CONFUSION MATRIX

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_test,
                        y_pred,
                        labels=[0, 1]
                    )

                    cm_df = pd.DataFrame(
                        cm,
                        index=[
                            "Actual Benign",
                            "Actual Malignant"
                        ],
                        columns=[
                            "Predicted Benign",
                            "Predicted Malignant"
                        ]
                    )

                    st.dataframe(cm_df)


                    # STEP 12: DISPLAY CLASSIFICATION REPORT

                    st.subheader("Classification Report")

                    report = classification_report(
                        y_test,
                        y_pred,
                        labels=[0, 1],
                        target_names=[
                            "Benign",
                            "Malignant"
                        ],
                        zero_division=0
                    )

                    st.text(report)


                    # STEP 13: DISPLAY SELECTED MODEL

                    st.success(
                        f"Evaluation completed successfully "
                        f"using {selected_model}!"
                    )


    except Exception as e:

        st.error(f"An error occurred: {e}")


# END OF APPLICATION

