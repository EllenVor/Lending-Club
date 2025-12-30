import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.data_modelling import train_model, evaluate_model

# Load datasets
@st.cache_data
def load_data():
    import os
    processed_dir = repo_root / "data" / "processed"
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
            df = pd.read_csv(processed_dir / file, index_col=False)
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

# Compare models
if st.sidebar.button("Compare Models"):
    df = datasets[selected_dataset]
    X = df.loc[:, df.columns != 'not.fully.paid']
    y = df['not.fully.paid']
    smote = ADASYN(sampling_strategy='minority')
    X, y = smote.fit_resample(X, y)
    
    results = []
    roc_data = {}
    conf_matrices = {}
    for model_name in models:
        model, feature_names, X_test, y_test = train_model(model_name, X, y)
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
        recall = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
        f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'ROC AUC': roc_auc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Store ROC data
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[model_name] = (fpr, tpr, roc_auc)
        
        # Store confusion matrix
        conf_matrices[model_name] = confusion_matrix(y_test, y_pred)
    
    results_df = pd.DataFrame(results)
    st.header("Model Comparison")
    st.table(results_df)
    
    # Confusion Matrices
    st.header("Confusion Matrices")
    cols = st.columns(3)
    for i, model_name in enumerate(models):
        with cols[i]:
            st.subheader(f"{model_name}")
            cm = conf_matrices[model_name]
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{model_name}')
            st.pyplot(fig)
    
    # ROC Curves
    st.header("ROC Curves")
    fig, ax = plt.subplots()
    for model_name in models:
        fpr, tpr, roc_auc = roc_data[model_name]
        ax.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
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
    
    # Train model on selected dataset (using Random Forest as default)
    df = datasets[selected_dataset]
    X = df.loc[:, df.columns != 'not.fully.paid']
    y = df['not.fully.paid']
    smote = ADASYN(sampling_strategy='minority')
    X, y = smote.fit_resample(X, y)
    model, _, _, _ = train_model("Random Forest", X, y)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("Prediction Results")
    st.write(f"Predicted Default: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Probability of Default: {probability:.4f}")
