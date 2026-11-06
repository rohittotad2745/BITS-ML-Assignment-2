import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    label_encoder_path = MODEL_DIR / "label_encoder.joblib"

    if not preprocessor_path.exists() or not label_encoder_path.exists():
        raise FileNotFoundError(
            "Missing preprocessor.joblib or label_encoder.joblib in the models/ directory."
        )

    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    models = {}
    missing = []
    for display_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[display_name] = joblib.load(path)
        else:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            "Missing model file(s) in models/: " + ", ".join(missing)
        )

    return preprocessor, label_encoder, models


def prepare_uploaded_data(uploaded_df, preprocessor, label_encoder):
    data = uploaded_df.copy()

    if "y" not in data.columns:
        raise ValueError(
            "The uploaded CSV must contain the actual target column 'y' "
            "with values 'yes' and 'no' so evaluation metrics can be calculated."
        )

    if data["y"].isna().any():
        raise ValueError("The target column 'y' contains missing values.")

    allowed_targets = set(label_encoder.classes_)
    observed_targets = set(data["y"].astype(str).unique())
    unexpected = observed_targets - allowed_targets
    if unexpected:
        raise ValueError(
            f"Unexpected target value(s): {sorted(unexpected)}. "
            f"Expected: {sorted(allowed_targets)}."
        )

    y_true = label_encoder.transform(data["y"].astype(str))

    # duration is deliberately excluded because it is unavailable before the call.
    X_raw = data.drop(columns=["y", "duration"], errors="ignore")

    expected_features = list(preprocessor.feature_names_in_)
    missing_features = sorted(set(expected_features) - set(X_raw.columns))
    extra_features = sorted(set(X_raw.columns) - set(expected_features))

    if missing_features:
        raise ValueError("Missing required feature(s): " + ", ".join(missing_features))

    if extra_features:
        st.warning(
            "Extra column(s) will be ignored: " + ", ".join(extra_features)
        )

    X_raw = X_raw[expected_features]

    # Ensure the uploaded CSV has the expected feature types.
    for col in X_raw.columns:
        if col in preprocessor.transformers_[0][2]:
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    if X_raw.isna().any().any():
        bad_columns = X_raw.columns[X_raw.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values found in required feature(s): "
            + ", ".join(bad_columns)
        )

    X_processed = preprocessor.transform(X_raw)
    return X_processed, y_true


st.title("📊 Bank Marketing Subscription Classifier")
st.write(
    "Upload a held-out test CSV to evaluate one of the five trained classification "
    "models. The application applies the saved preprocessing pipeline and does not "
    "retrain models."
)

st.info(
    "Upload test data only. The CSV must contain the 15 raw input features and "
    "the target column `y` (`yes`/`no`). The `duration` column is optional and "
    "will be excluded to prevent target leakage."
)

try:
    preprocessor, label_encoder, models = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load model artifacts: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

model_name = st.selectbox("Select classification model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file, sep=None, engine="python")
        st.subheader("Uploaded dataset preview")
        st.dataframe(uploaded_df.head(10), use_container_width=True)
        st.caption(f"Uploaded rows: {len(uploaded_df):,}")

        if st.button("Evaluate selected model", type="primary"):
            X_processed, y_true = prepare_uploaded_data(
                uploaded_df, preprocessor, label_encoder
            )
            model = models[model_name]
            y_pred = model.predict(X_processed)

            if not hasattr(model, "predict_proba"):
                st.error("The selected model does not provide probability estimates.")
                st.stop()

            positive_class_index = list(model.classes_).index(
                label_encoder.transform(["yes"])[0]
            )
            y_score = model.predict_proba(X_processed)[:, positive_class_index]

            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_score),
                "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                "F1 Score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
                "MCC": matthews_corrcoef(y_true, y_pred),
            }

            st.subheader(f"Evaluation results: {model_name}")
            metric_cols = st.columns(3)
            for index, (metric_name, value) in enumerate(metrics.items()):
                metric_cols[index % 3].metric(metric_name, f"{value:.4f}")

            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Predicted No", "Predicted Yes"],
                yticklabels=["Actual No", "Actual Yes"],
                ax=ax,
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification report")
            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["No", "Yes"],
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

            results = uploaded_df.copy()
            results["actual_y_encoded"] = y_true
            results["predicted_y_encoded"] = y_pred
            results["predicted_y"] = label_encoder.inverse_transform(y_pred)
            results["probability_yes"] = y_score

            st.download_button(
                "Download predictions CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name=f"{model_name.lower().replace(' ', '_')}_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Could not evaluate the uploaded CSV: {exc}")
else:
    st.caption("Choose a CSV file to begin evaluation.")
