import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, classification_report
)
from sklearn.preprocessing import label_binarize
from models.preprocessing import preprocess_data

def decision_tree(df):
    X, y, target_names = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))
    auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")

    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )

    classes = sorted(y.unique())                
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

    results = pd.DataFrame([{
        "Model": "Decision Tree",
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }])

    return results, report_dict, per_class_counts