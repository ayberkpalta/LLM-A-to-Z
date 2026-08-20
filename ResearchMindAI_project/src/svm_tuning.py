"""
ResearchMindAI
SVM Hyperparameter Tuning

Amaç:
MPNet embedding'leri üzerinde Linear SVM için
en iyi C hyperparameter'ını validation seti
kullanarak bulmak.

ÖNEMLİ:
Test seti bu aşamada kullanılmaz.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np

from pathlib import Path

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


# Denenecek C değerleri
C_VALUES = [
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0
]


RANDOM_STATE = 42


CLASSES = [
    "NLP",
    "Computer Vision",
    "Machine Learning",
    "Robotics"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(
        EMBEDDING_DIR /
        "train_embeddings_final.npy"
    )

    X_validation = np.load(
        EMBEDDING_DIR /
        "validation_embeddings_final.npy"
    )

    Y_train = np.load(
        LABEL_DIR /
        "Y_train_final.npy"
    )

    Y_validation = np.load(
        LABEL_DIR /
        "Y_validation_final.npy"
    )

    return (
        X_train,
        X_validation,
        Y_train,
        Y_validation
    )


# ============================================================
# TRAIN SVM
# ============================================================

def train_svm(
    X_train,
    Y_train,
    C
):

    classifier = OneVsRestClassifier(
        LinearSVC(
            C=C,
            random_state=RANDOM_STATE,
            max_iter=5000
        )
    )

    classifier.fit(
        X_train,
        Y_train
    )

    return classifier


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    classifier,
    X_validation,
    Y_validation
):

    predictions = classifier.predict(
        X_validation
    )

    micro_precision = precision_score(
        Y_validation,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        Y_validation,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        Y_validation,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        Y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        Y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        Y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — SVM HYPERPARAMETER TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        Y_train,
        Y_validation
    ) = load_data()

    print()
    print("DATA")
    print("-" * 70)

    print(
        f"X_train:      {X_train.shape}"
    )

    print(
        f"X_validation: {X_validation.shape}"
    )

    print(
        f"Y_train:      {Y_train.shape}"
    )

    print(
        f"Y_validation: {Y_validation.shape}"
    )

    print()
    print("Classes:")
    print(CLASSES)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    # ========================================================
    # TEST DIFFERENT C VALUES
    # ========================================================

    print()
    print("=" * 70)
    print("TESTING C VALUES")
    print("=" * 70)

    for C in C_VALUES:

        print()
        print(
            f"Training Linear SVM "
            f"with C={C}"
        )

        classifier = train_svm(
            X_train=X_train,
            Y_train=Y_train,
            C=C
        )

        metrics = evaluate_model(
            classifier=classifier,
            X_validation=X_validation,
            Y_validation=Y_validation
        )

        result = {
            "C": C,
            **metrics
        }

        results.append(result)

        print(
            f"Micro Precision: "
            f"{metrics['micro_precision']:.3f}"
        )

        print(
            f"Micro Recall:    "
            f"{metrics['micro_recall']:.3f}"
        )

        print(
            f"Micro F1:        "
            f"{metrics['micro_f1']:.3f}"
        )

        print(
            f"Macro Precision: "
            f"{metrics['macro_precision']:.3f}"
        )

        print(
            f"Macro Recall:    "
            f"{metrics['macro_recall']:.3f}"
        )

        print(
            f"Macro F1:        "
            f"{metrics['macro_f1']:.3f}"
        )

    # ========================================================
    # FIND BEST C
    # ========================================================

    best_result = max(
        results,
        key=lambda result: result["macro_f1"]
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("SVM HYPERPARAMETER TUNING RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'C':<10}"
        f"{'Micro F1':>12}"
        f"{'Macro F1':>12}"
        f"{'Precision':>14}"
        f"{'Recall':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['C']:<10}"
            f"{result['micro_f1']:>12.3f}"
            f"{result['macro_f1']:>12.3f}"
            f"{result['macro_precision']:>14.3f}"
            f"{result['macro_recall']:>12.3f}"
        )

    # ========================================================
    # BEST HYPERPARAMETER
    # ========================================================

    print()
    print("=" * 70)
    print("BEST SVM HYPERPARAMETER")
    print("=" * 70)

    print(
        f"Best C: "
        f"{best_result['C']}"
    )

    print(
        f"Validation Micro F1: "
        f"{best_result['micro_f1']:.3f}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_result['macro_f1']:.3f}"
    )

    print(
        f"Validation Macro Precision: "
        f"{best_result['macro_precision']:.3f}"
    )

    print(
        f"Validation Macro Recall: "
        f"{best_result['macro_recall']:.3f}"
    )

    print()
    print("=" * 70)
    print("SVM TUNING COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()