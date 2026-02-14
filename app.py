import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

st.set_page_config(
    page_title="ML Assignment 2 - Breast Cancer",
    layout="wide"
)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("Model Selection Panel")
st.sidebar.markdown("Upload dataset and choose a model to evaluate.")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

model_paths = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "Random Forest": "model/Random_Forest.pkl",
    "XGBoost": "model/XGBoost.pkl"
}

selected_model = st.sidebar.selectbox("Choose Model", list(model_paths.keys()))

# ---------------------------
# MAIN TITLE
# ---------------------------
st.title(" Breast Cancer Classification")
st.markdown("### Machine Learning Assignment 2 - M.Tech AIML")
st.markdown("---")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV Test File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader(" Dataset Preview")
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.head())

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
        y_true = df["diagnosis"]
        X = df.drop("diagnosis", axis=1)
    else:
        st.error("CSV must contain 'diagnosis' column.")
        st.stop()

    model = joblib.load(model_paths[selected_model])

    if selected_model in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)

    predictions = model.predict(X)

    st.markdown("---")
    st.subheader(f" Results using {selected_model}")

    # Accuracy Display
    accuracy = accuracy_score(y_true, predictions)
    col1, col2 = st.columns(2)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.write("")

    # Classification Report
    st.markdown("**Class Labels:**")
    st.markdown("- 0 → Benign Tumor")
    st.markdown("- 1 → Malignant Tumor")
    st.subheader(" Classification Report")
    st.text(classification_report(y_true, predictions))

    # Confusion Matrix
    st.subheader(" Confusion Matrix")
    cm = confusion_matrix(y_true, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax)
    st.pyplot(fig)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("Developed for ML Assignment 2 | BITS M.Tech AIML | 2025AA05089")