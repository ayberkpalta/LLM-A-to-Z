"""
ResearchMindAI
MPNet Embeddings + Linear SVM
"""

import numpy as np

from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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

CLASSES = [
    "NLP",
    "Computer Vision",
    "Machine Learning",
    "Robotics"
]

THRESHOLD = 0.0

RANDOM_STATE = 42


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

    return (
        X_train,
        X_test,
        Y_train,
        Y_test
    )


# ============================================================
# TRAIN SVM
# ============================================================

def train_classifier(
    X_train,
    Y_train
):

    classifier = OneVsRestClassifier(
        LinearSVC(
            C=1.0,
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
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    classifier,
    X_test
):

    decision_scores = classifier.decision_function(
        X_test
    )

    predictions = (
        decision_scores >= THRESHOLD
    ).astype(int)

    return predictions


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    Y_test,
    predictions
):

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

    print()
    print("=" * 70)
    print("SVM TEST RESULTS")
    print("=" * 70)

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

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            Y_test,
            predictions,
            target_names=CLASSES,
            zero_division=0
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — MPNet + LINEAR SVM")
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
    print(
        f"Train embeddings: {X_train.shape}"
    )

    print(
        f"Test embeddings:  {X_test.shape}"
    )

    print(
        f"Train labels:      {Y_train.shape}"
    )

    print(
        f"Test labels:       {Y_test.shape}"
    )

    print()
    print(
        f"Classes: {CLASSES}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("Training Linear SVM...")

    classifier = train_classifier(
        X_train,
        Y_train
    )

    print(
        "SVM classifier trained."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print()
    print(
        "Generating test predictions..."
    )

    predictions = generate_predictions(
        classifier,
        X_test
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

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluate_model(
        Y_test,
        predictions
    )

    print()
    print("=" * 70)
    print("SVM EVALUATION COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()