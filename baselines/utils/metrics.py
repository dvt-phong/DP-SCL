import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        threshold = 0.5
    else:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        threshold = float(thresholds[np.argmax(tpr - fpr)])

    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": threshold,
        "auc": float(auc),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def get_estimator_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return np.asarray(scores).reshape(-1)

    return np.asarray(model.predict(X_test)).reshape(-1)

