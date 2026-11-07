# BioDegrade Lab — QSAR Classification Benchmark

## a. Problem Statement

The objective of this project is to compare five machine-learning classification models for predicting whether a chemical compound is readily biodegradable using its molecular-descriptor values.

All five models were trained using the same training split and evaluated using the same held-out test split so that their performance could be compared consistently. The trained models were saved and are used by the Streamlit application for evaluation without retraining on the uploaded test data.

## b. Dataset Description

| Property | Details |
| --- | --- |
| Dataset | UCI QSAR Biodegradation |
| UCI Dataset ID | 254 |
| DOI | 10.24432/C5H60M |
| Task | Binary classification |
| Instances | 1,055 |
| Input features | 41 |
| Feature type | Numeric molecular descriptors |
| Target | Ready Biodegradable (RB = 1) vs Not Ready Biodegradable (NRB = 0) |
| Missing values | None |
| Train/Test split | 80% / 20% |
| Random seed | 5747 |
| UCI Source URL | https://archive.ics.uci.edu/dataset/254/qsar+biodegradation |

The dataset satisfies the assignment requirement of at least 500 instances and at least 12 input features.

## c. GitHub Repository Link

https://github.com/LakshminathKillana/qsar-biodegradation-2026aj05747

## Live Streamlit Application Link

https://qsar-biodegradation-2026aj05747.streamlit.app/

## d. Models Used and Evaluation Metrics

The following five models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

All models were evaluated using:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC Score)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8910 | 0.9392 | 0.8000 | 0.9014 | 0.8477 | 0.7666 |
| Decision Tree | 0.8531 | 0.9076 | 0.7564 | 0.8310 | 0.7919 | 0.6806 |
| K-Nearest Neighbors | 0.8768 | 0.9183 | 0.7848 | 0.8732 | 0.8267 | 0.7340 |
| Gaussian Naive Bayes | 0.6588 | 0.8076 | 0.4964 | 0.9577 | 0.6538 | 0.4603 |
| Support Vector Machine | 0.9005 | 0.9229 | 0.8289 | 0.8873 | 0.8571 | 0.7820 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Logistic Regression performed very well and achieved the highest AUC of 0.9392. Its recall was also strong, so it identified most of the ready-biodegradable samples. Overall, it was one of the strongest models in the comparison. |
| Decision Tree | Decision Tree gave reasonable results, but its accuracy, F1 score and MCC were lower than Logistic Regression, KNN and SVM. It performed adequately, but made more incorrect predictions than the stronger models. |
| K-Nearest Neighbors | KNN gave balanced results across accuracy, precision and recall. It performed better than Decision Tree and Gaussian Naive Bayes, but remained slightly below Logistic Regression and SVM. |
| Gaussian Naive Bayes | Gaussian Naive Bayes achieved the highest recall, so it detected almost all ready-biodegradable samples. However, its precision was low, meaning it also classified many non-biodegradable samples as biodegradable. This reduced its overall accuracy and MCC. |
| Support Vector Machine | SVM gave the best overall performance. It achieved the highest accuracy, precision, F1 score and MCC among the five models while also maintaining strong recall and AUC. |
| **Overall, Winner for your dataset?** | **Support Vector Machine.** Logistic Regression achieved a slightly higher AUC, but SVM provided the best overall balance because it achieved the highest Accuracy, Precision, F1 Score and MCC. |
