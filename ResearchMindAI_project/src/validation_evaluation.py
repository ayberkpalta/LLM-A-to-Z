"""
ResearchMindAI — Validation Evaluation

Eğitilmiş Multi-Label Logistic Regression
modelinin validation performansını ölçer.

1. Validation probability'lerini üretir.
2. Farklı threshold değerlerini dener.
3. Micro F1 ve Macro F1 hesaplar.
4. En iyi threshold'u belirler.
"""

import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_DIR = Path(
    "data/processed/embeddings"
)

LABEL_DIR = Path(
    "data/processed/labels"
)

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Embedding ve label dosyalarını yükler.
    """

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

    classes = np.load(
        LABEL_DIR / "classes_final.npy",
        allow_pickle=True
    )

    return (
        X_train,
        X_validation,
        Y_train,
        Y_validation,
        classes
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    Y_train
):
    """
    Train dataset üzerinde
    Multi-Label Logistic Regression eğitir.
    """

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
# VALIDATION PROBABILITIES
# ============================================================

def get_validation_probabilities(
    classifier,
    X_validation
):
    """
    Validation paper'ları için
    class probability değerlerini üretir.
    """

    probabilities = classifier.predict_proba(
        X_validation
    )

    return probabilities


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(
    probabilities,
    Y_validation
):
    """
    Farklı threshold değerlerini test eder.

    Her threshold için:

    Probability >= threshold
        → 1

    Probability < threshold
        → 0
    """

    best_threshold = None
    best_micro_f1 = -1

    print()
    print("=" * 70)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 70)

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        predicted_positives = (
            predictions.sum()
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Micro F1: {micro_f1:.3f} | "
            f"Macro F1: {macro_f1:.3f} | "
            f"Predicted Positives: "
            f"{predicted_positives}"
        )

        if micro_f1 > best_micro_f1:

            best_micro_f1 = micro_f1

            best_threshold = threshold

    return (
        best_threshold,
        best_micro_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — VALIDATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        Y_train,
        Y_validation,
        classes
    ) = load_data()

    print()
    print("Train embeddings:", X_train.shape)
    print("Validation embeddings:", X_validation.shape)

    print("Train labels:", Y_train.shape)
    print("Validation labels:", Y_validation.shape)

    print()
    print("Classes:", classes)

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
    print("Generating validation probabilities...")

    probabilities = get_validation_probabilities(
        classifier,
        X_validation
    )

    print(
        "Probability shape:",
        probabilities.shape
    )

    print()
    print("First 5 validation probabilities:")

    print(
        probabilities[:5]
    )

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    (
        best_threshold,
        best_micro_f1
    ) = evaluate_thresholds(
        probabilities,
        Y_validation
    )

    # --------------------------------------------------------
    # Final validation result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best Micro F1: "
        f"{best_micro_f1:.3f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
