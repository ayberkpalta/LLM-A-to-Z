"""
ResearchMindAI — Final Test Evaluation

Validation set üzerinde belirlenen threshold'u
kullanarak test setindeki final model performansını ölçer.
"""

import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
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

# Validation sonucunda seçtiğimiz threshold
BEST_THRESHOLD = 0.45


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(
        EMBEDDING_DIR / "train_embeddings_final.npy"
    )

    X_test = np.load(
        EMBEDDING_DIR / "test_embeddings_final.npy"
    )

    Y_train = np.load(
        LABEL_DIR / "Y_train_final.npy"
    )

    Y_test = np.load(
        LABEL_DIR / "Y_test_final.npy"
    )

    classes = np.load(
        LABEL_DIR / "classes_final.npy",
        allow_pickle=True
    )

    return (
        X_train,
        X_test,
        Y_train,
        Y_test,
        classes
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    Y_train
):

    base_classifier = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    classifier = OneVsRestClassifier(
        base_classifier
    )

    classifier.fit(
        X_train,
        Y_train
    )

    return classifier


# ============================================================
# TEST PROBABILITIES
# ============================================================

def get_test_probabilities(
    classifier,
    X_test
):

    probabilities = classifier.predict_proba(
        X_test
    )

    return probabilities


# ============================================================
# CONVERT PROBABILITIES TO PREDICTIONS
# ============================================================

def apply_threshold(
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    Y_test,
    predictions,
    classes
):

    micro_f1 = f1_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    micro_precision = precision_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    micro_recall = recall_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_recall = recall_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Threshold: {BEST_THRESHOLD}"
    )

    print()
    print(
        f"Micro Precision: {micro_precision:.3f}"
    )

    print(
        f"Micro Recall:    {micro_recall:.3f}"
    )

    print(
        f"Micro F1:        {micro_f1:.3f}"
    )

    print()

    print(
        f"Macro Precision: {macro_precision:.3f}"
    )

    print(
        f"Macro Recall:    {macro_recall:.3f}"
    )

    print(
        f"Macro F1:        {macro_f1:.3f}"
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            Y_test,
            predictions,
            target_names=classes,
            zero_division=0
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — FINAL TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        classes
    ) = load_data()

    print()
    print("Train embeddings:", X_train.shape)
    print("Test embeddings:", X_test.shape)

    print("Train labels:", Y_train.shape)
    print("Test labels:", Y_test.shape)

    print()
    print("Classes:", classes)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("Training classifier...")

    classifier = train_classifier(
        X_train,
        Y_train
    )

    print("Classifier trained.")

    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    print()
    print("Generating test probabilities...")

    probabilities = get_test_probabilities(
        classifier,
        X_test
    )

    print(
        "Probability shape:",
        probabilities.shape
    )

    print()
    print("First 5 test probabilities:")

    print(
        probabilities[:5]
    )

    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    predictions = apply_threshold(
        probabilities,
        BEST_THRESHOLD
    )

    print()
    print(
        "Predictions shape:",
        predictions.shape
    )

    print(
        "Total predicted positives:",
        predictions.sum()
    )

    print()
    print("First 5 predictions:")

    print(
        predictions[:5]
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_model(
        Y_test,
        predictions,
        classes
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()