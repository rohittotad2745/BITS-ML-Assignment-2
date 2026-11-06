# Machine Learning Assignment 2
## Bank Marketing Subscription Prediction

### 1. Project Overview

This project implements and compares five supervised machine learning classification algorithms to predict whether a customer will subscribe to a term deposit based on their banking and demographic information.

The project uses the **Bank Marketing Dataset from the UCI Machine Learning Repository** and evaluates multiple classification models using standard performance metrics.

The objective is to identify a suitable classification model and understand how different algorithms perform on the same dataset.

### 2. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**File used:** `bank.csv`  
**Total instances:** 4,521  
**Target variable:** `y`

The target variable represents whether a customer subscribed to a term deposit:

- `yes` – Customer subscribed.
- `no` – Customer did not subscribe.

The dataset contains demographic, financial, and campaign-related information.

**Feature Engineering:**

The `duration` feature was excluded from model training because call duration is only known after the marketing call. Including it would introduce target leakage when predicting subscription before a call takes place.

The remaining 15 input features are used for model training.

### 3. Machine Learning Models

The following five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models were trained using the same training dataset and evaluated using the same test dataset.

### 4. Data Preprocessing

The following preprocessing steps were performed:

- Separated input features and target variable.
- Excluded the `duration` feature to prevent target leakage.
- Encoded the target variable (`no` = 0, `yes` = 1).
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used stratified sampling to preserve the target class distribution.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Used a ColumnTransformer to combine the preprocessing steps.

The preprocessing pipeline was fitted only on the training dataset to avoid data leakage.

### 5. Model Evaluation

The models were evaluated using the following metrics:

| Metric | Description |
|---|
