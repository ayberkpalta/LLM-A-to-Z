"""
ResearchMindAI
Final SVM Test Evaluation

Validation tuning sonucunda seçilen:
C = 0.25

ile Linear SVM'i train setinde eğitir,
daha önce hiç kullanılmayan test setinde
final performansı ölçer.
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
    f1_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("data/processed")

EMBEDDING_DIR = BASE_DIR / "embeddings"
LABEL_DIR = BASE_DIR / "labels"


# Validation tuning sonucunda seçildi
BEST_C = 0.25

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
# TRAIN FINAL SVM
# ============================================================

def train_final_svm(
    X_train,
    Y_train
):

    classifier = OneVsRestClassifier(
        LinearSVC(
            C=BEST_C,
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
# EVALUATE
# ============================================================

def evaluate_model(
    classifier,
    X_test,
    Y_test
):

    predictions = classifier.predict(
        X_test
    )

    # --------------------------------------------------------
    # Micro Metrics
    # --------------------------------------------------------

    micro_precision = precision_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Macro Metrics
    # --------------------------------------------------------

    macro_precision = precision_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    return (
        predictions,
        micro_precision,
        micro_recall,
        micro_f1,
        macro_precision,
        macro_recall,
        macro_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — FINAL SVM TEST")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

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

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING FINAL SVM")
    print("=" * 70)

    print(
        f"C = {BEST_C}"
    )

    classifier = train_final_svm(
        X_train=X_train,
        Y_train=Y_train
    )

    print()
    print("Final SVM trained successfully.")

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    (
        predictions,
        micro_precision,
        micro_recall,
        micro_f1,
        macro_precision,
        macro_recall,
        macro_f1
    ) = evaluate_model(
        classifier=classifier,
        X_test=X_test,
        Y_test=Y_test
    )

    print(
        f"Prediction shape: "
        f"{predictions.shape}"
    )

    print(
        f"Total predicted positives: "
        f"{predictions.sum()}"
    )

    print()
    print("First 5 predictions:")

    print(
        predictions[:5]
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SVM TEST RESULTS")
    print("=" * 70)

    print()
    print("C")
    print("-" * 70)

    print(
        f"Best C: {BEST_C}"
    )

    print()
    print("MICRO METRICS")
    print("-" * 70)

    print(
        f"Precision: {micro_precision:.3f}"
    )

    print(
        f"Recall:    {micro_recall:.3f}"
    )

    print(
        f"F1:        {micro_f1:.3f}"
    )

    print()
    print("MACRO METRICS")
    print("-" * 70)

    print(
        f"Precision: {macro_precision:.3f}"
    )

    print(
        f"Recall:    {macro_recall:.3f}"
    )

    print(
        f"F1:        {macro_f1:.3f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        Y_test,
        predictions,
        target_names=CLASSES,
        zero_division=0
    )

    print(report)

    # ========================================================
    # COMPARISON WITH PREVIOUS SVM
    # ========================================================

    print("=" * 70)
    print("PREVIOUS vs TUNED SVM")
    print("=" * 70)

    previous_micro_f1 = 0.842
    previous_macro_f1 = 0.838

    print()
    print(
        f"Previous SVM (C=1.0) "
        f"Micro F1: {previous_micro_f1:.3f}"
    )

    print(
        f"Tuned SVM (C={BEST_C}) "
        f"Micro F1: {micro_f1:.3f}"
    )

    print()

    print(
        f"Previous SVM (C=1.0) "
        f"Macro F1: {previous_macro_f1:.3f}"
    )

    print(
        f"Tuned SVM (C={BEST_C}) "
        f"Macro F1: {macro_f1:.3f}"
    )

    print()
    print("=" * 70)
    print("FINAL SVM EVALUATION COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()