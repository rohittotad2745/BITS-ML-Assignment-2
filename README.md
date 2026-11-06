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
|---|---|
| Accuracy | Overall proportion of correct predictions |
| AUC | Ability of the model to distinguish between the two classes |
| Precision | Proportion of positive predictions that were correct |
| Recall | Proportion of actual positive cases correctly identified |
| F1 Score | Harmonic mean of precision and recall |
| MCC | Matthews Correlation Coefficient, considering all four confusion matrix outcomes |

#### Model Comparison Results

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8873 | 0.7376 | 0.5333 | 0.1538 | 0.2388 | 0.2429 |
| Decision Tree | 0.8088 | 0.5657 | 0.2149 | 0.2500 | 0.2311 | 0.1231 |
| KNN | 0.8762 | 0.6112 | 0.2778 | 0.0481 | 0.0820 | 0.0727 |
| Gaussian Naive Bayes | 0.8099 | 0.7166 | 0.2763 | 0.4038 | 0.3281 | 0.2274 |
| SVM | 0.8884 | 0.6909 | 0.5600 | 0.1346 | 0.2171 | 0.2352 |

#### Observations

- **Logistic Regression:** Achieved the highest AUC and MCC, indicating the strongest overall discrimination and correlation among the evaluated models.
- **Decision Tree:** Showed comparatively weaker performance, with the lowest AUC.
- **KNN:** Achieved very low recall, indicating that it missed most actual subscribers.
- **Gaussian Naive Bayes:** Achieved the highest recall and F1 score, making it useful when identifying potential subscribers is the primary objective.
- **SVM:** Achieved the highest accuracy and precision, but its recall was relatively low.

**Overall observation:** Logistic Regression provides the strongest overall performance based on AUC and MCC. However, Gaussian Naive Bayes may be preferable when identifying more potential subscribers is the priority. Model selection ultimately depends on the business objective.

### 6. Project Structure

```text
BITS-ML-Assignment-2/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── bank.csv
│   │
│   └── processed/
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── test_data.csv
│
├── models/
│   ├── preprocessor.joblib
│   ├── label_encoder.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_nb.joblib
│   └── svm.joblib
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_machine_learning_model_implementation.ipynb
│   ├── 05_model_evaluation_and_comparison.ipynb
│   └── 06_streamlit_application.ipynb
│
└── results/
    ├── model_metrics.csv
    └── visualizations/
```

### 7. How to Run the Application Locally

Follow these steps to run the Streamlit application on your local machine.

#### Prerequisites

- Python 3.9 or later.
- Git.
- pip package manager.

#### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/BITS-ML-Assignment-2.git
```

Navigate to the project directory:

```bash
cd BITS-ML-Assignment-2
```

#### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

#### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your default browser.

If it does not open automatically, navigate to:

```text
http://localhost:8501
```

### 8. Test Dataset: Where to Find It

A sample test dataset is included in the repository:

```text
data/processed/test_data.csv
```

**How to download it:**

1. Open the GitHub repository.
2. Navigate to `data/processed/`.
3. Open `test_data.csv`.
4. Click the **Download raw file** button.
5. Save the CSV file to your computer.

Alternatively, if you have cloned the repository, you can directly select:

```text
data/processed/test_data.csv
```

from your local project directory.

**Important:** Upload `test_data.csv` to the Streamlit application. Do not upload `test_processed.csv`.

The `test_data.csv` file contains the original test features and actual target column (`y`), which are required to calculate the evaluation metrics.

### 9. Using the Streamlit Application

Once the application is running:

1. Upload `test_data.csv` using the CSV file uploader.
2. Select one of the five classification models from the model dropdown.
3. Execute the prediction and evaluation.
4. Review the evaluation metrics:
   - Accuracy
   - AUC
   - Precision
   - Recall
   - F1 Score
   - MCC
5. Review the confusion matrix or classification report to understand the model's predictions.

The application allows users to evaluate different classifiers using the same test dataset.

### 10. Live Application

**Streamlit Application:**  
[Add your deployed Streamlit URL here]

**GitHub Repository:**  
[Add your GitHub repository URL here]

### 11. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

### 12. Conclusion

This project demonstrates the implementation and comparative evaluation of five supervised machine learning classification algorithms using the Bank Marketing dataset.

The results highlight the importance of evaluating classification models using multiple metrics rather than relying solely on accuracy, particularly when working with imbalanced datasets.

Logistic Regression achieved the highest AUC and MCC, while Gaussian Naive Bayes achieved the highest recall and F1 score. These findings demonstrate how model selection should be guided by the intended business objective.
