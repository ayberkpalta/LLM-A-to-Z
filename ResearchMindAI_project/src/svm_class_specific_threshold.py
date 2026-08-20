"""
ResearchMindAI
SVM + Class-Specific Decision Threshold

Amaç:
Her sınıf için SVM decision score üzerinde
en iyi threshold değerini validation setinden bulmak.

Daha sonra bulunan threshold'ları
test setine uygulayarak final performansı ölçmek.
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

CLASSES = [
    "NLP",
    "Computer Vision",
    "Machine Learning",
    "Robotics"
]

# Validation tuning sonucunda seçtiğimiz C
BEST_C = 0.25

RANDOM_STATE = 42

# Decision score threshold adayları
THRESHOLDS = np.arange(
    -1.0,
    1.01,
    0.05
)


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

    X_test = np.load(
        EMBEDDING_DIR /
        "test_embeddings_final.npy"
    )

    Y_train = np.load(
        LABEL_DIR /
        "Y_train_final.npy"
    )

    Y_validation = np.load(
        LABEL_DIR /
        "Y_validation_final.npy"
    )

    Y_test = np.load(
        LABEL_DIR /
        "Y_test_final.npy"
    )

    return (
        X_train,
        X_validation,
        X_test,
        Y_train,
        Y_validation,
        Y_test
    )


# ============================================================
# TRAIN SVM
# ============================================================

def train_svm(
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
# FIND BEST THRESHOLD FOR ONE CLASS
# ============================================================

def find_best_threshold(
    y_true,
    scores
):

    best_threshold = 0.0
    best_f1 = -1.0

    for threshold in THRESHOLDS:

        predictions = (
            scores >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return (
        best_threshold,
        best_f1
    )


# ============================================================
# FIND CLASS-SPECIFIC THRESHOLDS
# ============================================================

def find_class_thresholds(
    validation_scores,
    Y_validation
):

    thresholds = []

    print()
    print("=" * 70)
    print("CLASS-SPECIFIC SVM THRESHOLD ANALYSIS")
    print("=" * 70)

    for class_index, class_name in enumerate(CLASSES):

        y_true = Y_validation[
            :,
            class_index
        ]

        scores = validation_scores[
            :,
            class_index
        ]

        (
            best_threshold,
            best_f1
        ) = find_best_threshold(
            y_true,
            scores
        )

        thresholds.append(
            best_threshold
        )

        print()
        print(class_name)
        print("-" * 70)

        print(
            f"Best threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Validation F1: "
            f"{best_f1:.3f}"
        )

    return np.array(thresholds)


# ============================================================
# APPLY CLASS-SPECIFIC THRESHOLDS
# ============================================================

def apply_thresholds(
    scores,
    thresholds
):

    predictions = np.zeros_like(
        scores,
        dtype=int
    )

    for class_index in range(
        scores.shape[1]
    ):

        predictions[
            :,
            class_index
        ] = (
            scores[
                :,
                class_index
            ]
            >= thresholds[class_index]
        ).astype(int)

    return predictions


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    Y_true,
    predictions
):

    micro_precision = precision_score(
        Y_true,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_recall = recall_score(
        Y_true,
        predictions,
        average="micro",
        zero_division=0
    )

    micro_f1 = f1_score(
        Y_true,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        Y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        Y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        Y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    return (
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
    print(
        "RESEARCHMINDAI — "
        "SVM CLASS-SPECIFIC THRESHOLD"
    )
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    (
        X_train,
        X_validation,
        X_test,
        Y_train,
        Y_validation,
        Y_test
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
        f"X_test:       {X_test.shape}"
    )

    print(
        f"Y_train:      {Y_train.shape}"
    )

    print(
        f"Y_validation: {Y_validation.shape}"
    )

    print(
        f"Y_test:       {Y_test.shape}"
    )

    print()
    print("Classes:")
    print(CLASSES)

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING SVM")
    print("=" * 70)

    print(
        f"C = {BEST_C}"
    )

    classifier = train_svm(
        X_train,
        Y_train
    )

    print(
        "SVM trained successfully."
    )

    # ========================================================
    # VALIDATION SCORES
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING VALIDATION SCORES")
    print("=" * 70)

    validation_scores = (
        classifier.decision_function(
            X_validation
        )
    )

    print(
        f"Validation score shape: "
        f"{validation_scores.shape}"
    )

    # ========================================================
    # FIND BEST THRESHOLDS
    # ========================================================

    class_thresholds = (
        find_class_thresholds(
            validation_scores,
            Y_validation
        )
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    validation_predictions = (
        apply_thresholds(
            validation_scores,
            class_thresholds
        )
    )

    (
        val_micro_precision,
        val_micro_recall,
        val_micro_f1,
        val_macro_precision,
        val_macro_recall,
        val_macro_f1
    ) = evaluate(
        Y_validation,
        validation_predictions
    )

    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print()
    print("Class-Specific Thresholds")
    print("-" * 70)

    for index, class_name in enumerate(CLASSES):

        print(
            f"{class_name}: "
            f"{class_thresholds[index]:.2f}"
        )

    print()
    print(
        f"Micro Precision: "
        f"{val_micro_precision:.3f}"
    )

    print(
        f"Micro Recall:    "
        f"{val_micro_recall:.3f}"
    )

    print(
        f"Micro F1:        "
        f"{val_micro_f1:.3f}"
    )

    print(
        f"Macro Precision: "
        f"{val_macro_precision:.3f}"
    )

    print(
        f"Macro Recall:    "
        f"{val_macro_recall:.3f}"
    )

    print(
        f"Macro F1:        "
        f"{val_macro_f1:.3f}"
    )

    # ========================================================
    # TEST SCORES
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING TEST SCORES")
    print("=" * 70)

    test_scores = (
        classifier.decision_function(
            X_test
        )
    )

    print(
        f"Test score shape: "
        f"{test_scores.shape}"
    )

    # ========================================================
    # APPLY VALIDATION THRESHOLDS TO TEST
    # ========================================================

    test_predictions = (
        apply_thresholds(
            test_scores,
            class_thresholds
        )
    )

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    (
        test_micro_precision,
        test_micro_recall,
        test_micro_f1,
        test_macro_precision,
        test_macro_recall,
        test_macro_f1
    ) = evaluate(
        Y_test,
        test_predictions
    )

    # ========================================================
    # FINAL TEST RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print()
    print("CLASS-SPECIFIC THRESHOLDS")
    print("-" * 70)

    for index, class_name in enumerate(CLASSES):

        print(
            f"{class_name}: "
            f"{class_thresholds[index]:.2f}"
        )

    print()
    print("MICRO METRICS")
    print("-" * 70)

    print(
        f"Precision: "
        f"{test_micro_precision:.3f}"
    )

    print(
        f"Recall:    "
        f"{test_micro_recall:.3f}"
    )

    print(
        f"F1:        "
        f"{test_micro_f1:.3f}"
    )

    print()
    print("MACRO METRICS")
    print("-" * 70)

    print(
        f"Precision: "
        f"{test_macro_precision:.3f}"
    )

    print(
        f"Recall:    "
        f"{test_macro_recall:.3f}"
    )

    print(
        f"F1:        "
        f"{test_macro_f1:.3f}"
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
        test_predictions,
        target_names=CLASSES,
        zero_division=0
    )

    print(report)

    # ========================================================
    # COMPARISON
    # ========================================================

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print()
    print(
        "Previous SVM "
        "(C=0.25, default threshold)"
    )

    print(
        "Micro F1: 0.848"
    )

    print(
        "Macro F1: 0.843"
    )

    print()
    print(
        "SVM + Class-Specific Threshold"
    )

    print(
        f"Micro F1: "
        f"{test_micro_f1:.3f}"
    )

    print(
        f"Macro F1: "
        f"{test_macro_f1:.3f}"
    )

    print()
    print("=" * 70)
    print(
        "SVM CLASS-SPECIFIC "
        "THRESHOLD ANALYSIS COMPLETED"
    )
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()