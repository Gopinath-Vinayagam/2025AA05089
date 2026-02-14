# Machine Learning Assignment 2
## Breast Cancer Classification using Multiple ML Models

### Student ID: 2025AA05089
### Program: M.Tech (AIML)

--------------------------------------------------------------------------------------------

##  Problem Statement

The objective of this project is to classify breast cancer tumors as Malignant (M) or Benign (B) using multiple machine learning classification models. The goal is to evaluate and compare model performance using various evaluation metrics and deploy the solution as an interactive Streamlit web application.

--------------------------------------------------------------------------------------------

## . Dataset Description

Dataset Used: Breast Cancer Wisconsin Diagnostic Dataset

- Total Instances: 569
- Total Features: 30 numerical features
- Target Variable: Diagnosis (M = Malignant, B = Benign)
- Source: Kaggle Machine Learning Repository (https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?resource=download)

The dataset contains measurements computed from digitized images of fine needle aspirate (FNA) of breast mass tissue as per the dataset description.

Class Encoding:
0 → Benign
1 → Malignant

--------------------------------------------------------------------------------------------

##  Models Used

The following six classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Model Performance Comparison

| Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9123 | 0.9008 | 0.9000 | 0.8571 | 0.8780 | 0.8102 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9737 | 0.9950 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost | 0.9737 | 0.9940 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

--------------------------------------------------------------------------------------------

## Observations

- Random Forest and XGBoost achieved the highest accuracy and MCC score.
- Decision Tree showed comparatively lower performance due to overfitting tendency.
- Logistic Regression performed very well, indicating linear separability in data.
- Naive Bayes achieved high precision but slightly lower recall value.
- Ensemble models generally performed better than individual classifiers.

--------------------------------------------------------------------------------------------

## Streamlit App Features

- Upload CSV dataset
- Model selection dropdown
- Display of classification report
- Confusion matrix visualization
- Interactive frontend

---

## Deployment

The application is deployed using Streamlit Community Cloud.