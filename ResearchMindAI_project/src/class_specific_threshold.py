"""
ResearchMindAI
Class-Specific Threshold Optimization

Amaç:
Her sınıf için aynı threshold yerine,
validation setinde her sınıfa özel en iyi threshold'u bulmak.
"""

import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_DIR = Path(
    "data/processed/embeddings"
)

LABEL_DIR = Path(
    "data/processed/labels"
)

CLASSES = [
    "NLP",
    "Computer Vision",
    "Machine Learning",
    "Robotics"
]

RANDOM_STATE = 42

# Validation üzerinde denenecek threshold değerleri
THRESHOLDS = np.arange(
    0.20,
    0.56,
    0.05
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(
        EMBEDDING_DIR / "train_embeddings_final.npy"
    )

    X_validation = np.load(
        EMBEDDING_DIR / "validation_embeddings_final.npy"
    )

    Y_train = np.load(
        LABEL_DIR / "Y_train_final.npy"
    )

    Y_validation = np.load(
        LABEL_DIR / "Y_validation_final.npy"
    )

    return (
        X_train,
        X_validation,
        Y_train,
        Y_validation
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(X_train, Y_train):

    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )

    classifier.fit(
        X_train,
        Y_train
    )

    return classifier


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_thresholds(
    probabilities,
    Y_validation
):

    best_thresholds = {}

    print()
    print("=" * 70)
    print("CLASS-SPECIFIC THRESHOLD ANALYSIS")
    print("=" * 70)

    for class_index, class_name in enumerate(CLASSES):

        print()
        print(class_name)
        print("-" * 70)

        best_threshold = 0.45
        best_f1 = 0.0

        for threshold in THRESHOLDS:

            predictions = (
                probabilities[:, class_index] >= threshold
            ).astype(int)

            precision = precision_score(
                Y_validation[:, class_index],
                predictions,
                zero_division=0
            )

            recall = recall_score(
                Y_validation[:, class_index],
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                Y_validation[:, class_index],
                predictions,
                zero_division=0
            )

            print(
                f"Threshold: {threshold:.2f} | "
                f"Precision: {precision:.3f} | "
                f"Recall: {recall:.3f} | "
                f"F1: {f1:.3f}"
            )

            if f1 > best_f1:

                best_f1 = f1
                best_threshold = threshold

        best_thresholds[class_name] = best_threshold

        print()
        print(
            f"BEST THRESHOLD → {class_name}: "
            f"{best_threshold:.2f}"
        )

    return best_thresholds


# ============================================================
# CREATE PREDICTIONS
# ============================================================

def create_predictions(
    probabilities,
    thresholds
):

    predictions = np.zeros_like(
        probabilities,
        dtype=int
    )

    for class_index, class_name in enumerate(CLASSES):

        threshold = thresholds[class_name]

        predictions[:, class_index] = (
            probabilities[:, class_index] >= threshold
        ).astype(int)

    return predictions


# ============================================================
# EVALUATE PREDICTIONS
# ============================================================

def evaluate_predictions(
    Y_validation,
    predictions,
    name
):

    micro_f1 = f1_score(
        Y_validation,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        Y_validation,
        predictions,
        average="macro",
        zero_division=0
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

    print()
    print(name)
    print("-" * 70)

    print(
        f"Micro Precision: {micro_precision:.3f}"
    )

    print(
        f"Micro Recall:    {micro_recall:.3f}"
    )

    print(
        f"Micro F1:        {micro_f1:.3f}"
    )

    print(
        f"Macro F1:        {macro_f1:.3f}"
    )

    return micro_f1, macro_f1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — CLASS-SPECIFIC THRESHOLD")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        Y_train,
        Y_validation
    ) = load_data()

    print()
    print(
        f"Train embeddings:      {X_train.shape}"
    )

    print(
        f"Validation embeddings: {X_validation.shape}"
    )

    print(
        f"Train labels:          {Y_train.shape}"
    )

    print(
        f"Validation labels:     {Y_validation.shape}"
    )

    print()
    print(
        f"Classes: {CLASSES}"
    )

    # --------------------------------------------------------
    # Train classifier
    # --------------------------------------------------------

    print()
    print("Training classifier...")

    classifier = train_classifier(
        X_train,
        Y_train
    )

    print("Classifier trained.")

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print()
    print(
        "Generating validation probabilities..."
    )

    probabilities = classifier.predict_proba(
        X_validation
    )

    print(
        f"Probability shape: {probabilities.shape}"
    )

    # --------------------------------------------------------
    # Find best threshold for each class
    # --------------------------------------------------------

    best_thresholds = find_best_thresholds(
        probabilities,
        Y_validation
    )

    # --------------------------------------------------------
    # Global threshold = 0.45
    # --------------------------------------------------------

    global_threshold = 0.45

    global_predictions = (
        probabilities >= global_threshold
    ).astype(int)

    global_micro_f1, global_macro_f1 = evaluate_predictions(
        Y_validation,
        global_predictions,
        "GLOBAL THRESHOLD = 0.45"
    )

    # --------------------------------------------------------
    # Class-specific thresholds
    # --------------------------------------------------------

    class_specific_predictions = create_predictions(
        probabilities,
        best_thresholds
    )

    specific_micro_f1, specific_macro_f1 = evaluate_predictions(
        Y_validation,
        class_specific_predictions,
        "CLASS-SPECIFIC THRESHOLDS"
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    print()
    print("Global thresholds:")

    for class_name in CLASSES:

        print(
            f"{class_name}: 0.45"
        )

    print()
    print("Class-specific thresholds:")

    for class_name in CLASSES:

        print(
            f"{class_name}: "
            f"{best_thresholds[class_name]:.2f}"
        )

    print()
    print(
        f"Global Micro F1: "
        f"{global_micro_f1:.3f}"
    )

    print(
        f"Class-Specific Micro F1: "
        f"{specific_micro_f1:.3f}"
    )

    print()
    print(
        f"Global Macro F1: "
        f"{global_macro_f1:.3f}"
    )

    print(
        f"Class-Specific Macro F1: "
        f"{specific_macro_f1:.3f}"
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()