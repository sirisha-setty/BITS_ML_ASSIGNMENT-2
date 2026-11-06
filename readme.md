# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop a machine learning-based classification system to predict whether a breast tumour is Benign or Malignant using diagnostic features.

Five classification algorithms are implemented, evaluated and compared. An interactive Streamlit web application is developed to allow users to upload test data, select a trained model and view prediction results and evaluation metrics.

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

- Total instances: 569
- Total input features: 30
- Target variable: Diagnosis
- Classification type: Binary classification
- Benign cases: 357 (62.74%)
- Malignant cases: 212 (37.26%)

The dataset contains numerical features computed from digitized images of fine needle aspirates of breast masses.

The target variable is mapped as:
- 0 – Benign
- 1 – Malignant

## 3. GitHub Repository Link

https://github.com/sirisha-setty/BITS_ML_ASSIGNMENT_2

The repository contains:
- Python source code
- Dataset files
- Five trained machine learning models
- Preprocessing artifacts
- Streamlit application
- requirements.txt
- Test dataset

## 4. Models Used and Performance Comparison

The following five classification algorithms were implemented and evaluated using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on Model Performance

**Logistic Regression:** Achieved high accuracy and the highest AUC score of 0.9960, indicating excellent discrimination between the two classes.

**Decision Tree:** Provided reasonable classification performance but achieved comparatively lower accuracy and MCC.

**K-Nearest Neighbors:** Demonstrated strong classification performance with high accuracy and precision.

**Gaussian Naive Bayes:** Achieved perfect precision but comparatively lower recall, indicating that some malignant cases were not identified.

**Support Vector Machine:** Achieved the highest accuracy of 0.9737, perfect precision, and the highest MCC of 0.9442 among the evaluated models.

### Overall Winner

Support Vector Machine (SVM) is the best-performing model based on the highest accuracy, F1 score and Matthews Correlation Coefficient.

Logistic Regression achieved the highest AUC score.

## 5. Streamlit Application

The project includes an interactive web application developed using Streamlit.

### Application Features

- Upload test dataset in CSV format.
- Select one of the five trained classification models.
- Generate predictions using saved models.
- Display actual and predicted diagnosis.
- Display malignant prediction probabilities.
- Calculate Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- Display confusion matrix.
- Display classification report.

The application loads previously trained models and does not retrain models using uploaded test data.

### Live Streamlit Application

[PASTE YOUR DEPLOYED STREAMLIT APP URL HERE]

## 6. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

## 7. Conclusion

This project demonstrates an end-to-end machine learning classification workflow, including dataset exploration, preprocessing, model development, performance evaluation, model saving, application development and cloud deployment.

Among the five evaluated classifiers, Support Vector Machine achieved the best overall performance based on accuracy, F1 score and MCC.

The deployed Streamlit application provides an interactive interface for testing the trained models and evaluating their performance.

