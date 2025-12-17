import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from imblearn.over_sampling import SMOTE, ADASYN

def train_model(model_type, df, target_column):
    if target_column not in df.columns:
        raise ValueError("Target column is missing in dataset")
    if 'purpose' in df.columns:
        df = df.drop('purpose', axis=1)

    X = df.loc[:, df.columns != target_column]
    y = df[target_column]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = ADASYN(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)

    if model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000,class_weight='balanced', random_state=42)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_type == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_type == 'SVC':
        model = SVC()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.fit(X_train, y_train)
    return model, feature_names, X_test, y_test

def evaluate_model(model, X_test, y_test):
    A = 600
    B = 50
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    credit_score = A-B*np.log(y_proba/(1-y_proba)) # Example credit score formula

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    print(class_report)
    print(conf_matrix)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, roc_auc, class_report, conf_matrix

def computeFeatureOutliers(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the fraction (percentage) of outliers per feature.

    :param data_df: Input feature data
    :return: Table of computed outlier percentages
    """
    feature_names = []
    outlier_percentages = []

    for k, v in data_df.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        iqr = q3 - q1
        v_col = v[(v <= q1 - 1.5 * iqr) | (v >= q3 + 1.5 * iqr)]
        outlier_percentage = np.shape(v_col)[0] * 100.0 / np.shape(data_df)[0]
        feature_names.append(k)
        outlier_percentages.append(outlier_percentage)

    results = {
        "feature_names": feature_names,
        "outlier_percentages": outlier_percentages
    }

    results_df = pd.DataFrame(results)

    return results_df

def LinearReg(df, target_column):
    X = df.loc[:, df.columns != target_column]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lm = LinearRegression()
    lm.fit(X_train,y_train)

    predictions = lm.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, predictions))
    print('MSE:', mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))

    return lm