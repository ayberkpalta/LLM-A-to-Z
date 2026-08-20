"""
ResearchMindAI
Model Comparison

MPNet embeddings üzerinde farklı classifier'ların
test performanslarını karşılaştırır.

Models:
1. Logistic Regression
2. Logistic Regression + Class-Specific Threshold
3. Linear SVM
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("data/processed")

EMBEDDING_DIR = BASE_DIR / "embeddings"

LABEL_DIR = BASE_DIR / "labels"

TEST_PAPERS_PATH = (
    BASE_DIR / "test_papers_final.json"
)

CLASSES = [
    "NLP",
    "Computer Vision",
    "Machine Learning",
    "Robotics"
]


# Class-specific thresholds
CLASS_THRESHOLDS = np.array([
    0.40,   # NLP
    0.45,   # Computer Vision
    0.25,   # Machine Learning
    0.40    # Robotics
])


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(
        EMBEDDING_DIR /
        "train_embeddings_final.npy"
    )

    X_test = np.load(
        EMBEDDING_DIR /
        "test_embeddings_final.npy"
    )

    Y_train = np.load(
        LABEL_DIR /
        "Y_train_final.npy"
    )

    Y_test = np.load(
        LABEL_DIR /
        "Y_test_final.npy"
    )

    return (
        X_train,
        X_test,
        Y_train,
        Y_test
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    y_pred
):

    micro_precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "model": name,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X_train,
    Y_train,
    X_test
):

    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )

    classifier.fit(
        X_train,
        Y_train
    )

    probabilities = classifier.predict_proba(
        X_test
    )

    return probabilities


# ============================================================
# LINEAR SVM
# ============================================================

def train_svm(
    X_train,
    Y_train,
    X_test
):

    classifier = OneVsRestClassifier(
        LinearSVC(
            C=1.0,
            random_state=42
        )
    )

    classifier.fit(
        X_train,
        Y_train
    )

    predictions = classifier.predict(
        X_test
    )

    return predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — MODEL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        Y_train,
        Y_test
    ) = load_data()

    print()
    print("DATA")
    print("-" * 70)

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"Y_train: {Y_train.shape}"
    )

    print(
        f"Y_test:  {Y_test.shape}"
    )

    print()
    print("Classes:")
    print(CLASSES)

    results = []

    # ========================================================
    # MODEL 1
    # LOGISTIC REGRESSION — GLOBAL THRESHOLD
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL 1 — LOGISTIC REGRESSION")
    print("=" * 70)

    probabilities = train_logistic_regression(
        X_train,
        Y_train,
        X_test
    )

    global_threshold = 0.45

    predictions_global = (
        probabilities >= global_threshold
    ).astype(int)

    result = evaluate_model(
        "Logistic Regression (Global 0.45)",
        Y_test,
        predictions_global
    )

    results.append(result)

    # ========================================================
    # MODEL 2
    # LOGISTIC REGRESSION — CLASS SPECIFIC
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL 2 — LOGISTIC REGRESSION + CLASS-SPECIFIC")
    print("=" * 70)

    predictions_class_specific = (
        probabilities >= CLASS_THRESHOLDS
    ).astype(int)

    result = evaluate_model(
        "Logistic Regression (Class-Specific)",
        Y_test,
        predictions_class_specific
    )

    results.append(result)

    # ========================================================
    # MODEL 3
    # LINEAR SVM
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL 3 — LINEAR SVM")
    print("=" * 70)

    svm_predictions = train_svm(
        X_train,
        Y_train,
        X_test
    )

    result = evaluate_model(
        "Linear SVM",
        Y_test,
        svm_predictions
    )

    results.append(result)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Model':<40}"
        f"{'Micro F1':>12}"
        f"{'Macro F1':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['model']:<40}"
            f"{result['micro_f1']:>12.3f}"
            f"{result['macro_f1']:>12.3f}"
        )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model = max(
        results,
        key=lambda x: x["macro_f1"]
    )

    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Model: {best_model['model']}"
    )

    print(
        f"Micro F1: "
        f"{best_model['micro_f1']:.3f}"
    )

    print(
        f"Macro F1: "
        f"{best_model['macro_f1']:.3f}"
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()