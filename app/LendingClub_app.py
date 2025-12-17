import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score

# Add repo root to path
repo_root = Path.cwd().resolve().parent
sys.path.append(str(repo_root))

from src.data_modelling import train_model, evaluate_model

# Load datasets
@st.cache_data
def load_data():
    import os
    processed_dir = "../data/processed"
    datasets = {}
    name_mapping = {
        "loans_processed.csv": "Main Dataset",
        "loans_home_improvement_processed.csv": "Home Improvement",
        "loans_small_business_processed.csv": "Small Business",
        "loans_credit_card_processed.csv": "Credit Card",
        "loans_debt_consolidation_processed.csv": "Debt Consolidation",
        "loans_educational_processed.csv": "Educational",
        "loans_major_purchase_processed.csv": "Major Purchase",
        "loans_all_other_processed.csv": "All Other"
    }
    for file in os.listdir(processed_dir):
        if file.endswith("_processed.csv"):
            df = pd.read_csv(os.path.join(processed_dir, file), index_col=False)
            if 'purpose' in df.columns:
                df = df.drop(columns=['purpose'])
            readable_name = name_mapping.get(file, file.replace("_processed.csv", "").replace("loans_", "").replace("_", " ").title())
            datasets[readable_name] = df
    return datasets

datasets = load_data()

# Model options
models = ["Logistic Regression", "Random Forest", "XGBoost"]

st.title("Lending Club Loan Default Prediction")

# Sidebar
st.sidebar.header("Model Configuration")
selected_dataset = st.sidebar.radio("Select Dataset", list(datasets.keys()))
selected_model = st.sidebar.radio("Select Model", models)

# Train and evaluate model
if st.sidebar.button("Train and Evaluate Model"):
    df = datasets[selected_dataset]
    model, feature_names, X_test, y_test = train_model(selected_model, df, 'not.fully.paid')
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    st.header("Model Evaluation")
    
    st.subheader("Classification Report")
    st.text(class_report)
    
    st.subheader("Confusion Matrix")
    st.write(conf_matrix)
    
    st.subheader("Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"ROC AUC: {roc_auc:.4f}")
    
    # ROC Curve
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# New Client Prediction
st.header("Predict for New Client")
st.write("Enter client data below:")

# Get feature names from selected dataset (excluding target)
df_selected = datasets[selected_dataset]
feature_cols = [col for col in df_selected.columns if col != 'not.fully.paid']

# Create input fields
input_data = {}
for col in feature_cols:
    if df_selected[col].dtype == 'object':
        unique_vals = df_selected[col].unique()
        input_data[col] = st.selectbox(f"{col}", unique_vals)
    else:
        min_val = float(df_selected[col].min())
        max_val = float(df_selected[col].max())
        input_data[col] = st.slider(f"{col}", min_val, max_val, float(df_selected[col].mean()))

if st.button("Predict"):
    # Prepare input data
    input_df = pd.DataFrame([input_data])
    
    # Train model on selected dataset
    df = datasets[selected_dataset]
    model, _, _, _ = train_model(selected_model, df, 'not.fully.paid')
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("Prediction Results")
    st.write(f"Predicted Default: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Probability of Default: {probability:.4f}")
