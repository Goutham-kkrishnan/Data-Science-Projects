import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report
)
from sklearn.preprocessing import label_binarize
import pickle
import os
from models.preprocessing import preprocess_data

def naive_bayes(df):
    """Train and evaluate Gaussian Naive Bayes classifier."""
    # Preprocess
    X, y, target_names = preprocess_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale (GaussianNB can benefit from scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    # AUC
    y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))
    auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')

    # Classification report
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )

    # Save model and scaler
    os.makedirs("models", exist_ok=True)

    # Return results row
    results = pd.DataFrame([{
        'Model': 'Naive Bayes',
        'Accuracy': acc,
        'AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'MCC': mcc
    }])

    return results, report_dict