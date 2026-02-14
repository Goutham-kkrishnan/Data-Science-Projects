import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report
)
from sklearn.preprocessing import label_binarize
import pickle
import os
from models.preprocessing import preprocess_data

def knn(df):
    X, y, target_names = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=7, weights='distance')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))
    auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')

    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )

    # --- NEW: Per‑class correct / incorrect counts ---
    classes = sorted(y.unique())                # integer labels
    per_class_counts = {}
    for i, name in zip(classes, target_names):
        mask_true = (y_test == i)
        total = mask_true.sum()
        correct = ((y_test == i) & (y_pred == i)).sum()
        incorrect = total - correct
        per_class_counts[name] = {
            "correct": int(correct),
            "incorrect": int(incorrect),
            "total": int(total)
        }

    os.makedirs("models", exist_ok=True)

    results = pd.DataFrame([{
        'Model': 'KNN',
        'Accuracy': acc,
        'AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'MCC': mcc
    }])

    # Return three objects: results, report_dict, per_class_counts
    return results, report_dict, per_class_counts