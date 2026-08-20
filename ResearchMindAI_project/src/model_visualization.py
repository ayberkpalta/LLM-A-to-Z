"""
ResearchMindAI
Model Performance Visualization
"""

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
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

THRESHOLD = 0.45

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
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    Y_train
):

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
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    classifier,
    X_test
):

    probabilities = classifier.predict_proba(
        X_test
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return predictions


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(
    Y_test,
    predictions
):

    for class_index, class_name in enumerate(
        CLASSES
    ):

        true_labels = Y_test[
            :, class_index
        ]

        predicted_labels = predictions[
            :, class_index
        ]

        matrix = confusion_matrix(
            true_labels,
            predicted_labels
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[
                "Not " + class_name,
                class_name
            ]
        )

        display.plot()

        plt.title(
            f"Confusion Matrix — {class_name}"
        )

        plt.tight_layout()

        plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — MODEL VISUALIZATION")
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
        f"Test embeddings: {X_test.shape}"
    )

    print(
        f"Train labels: {Y_train.shape}"
    )

    print(
        f"Test labels: {Y_test.shape}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("Training classifier...")

    classifier = train_classifier(
        X_train,
        Y_train
    )

    print(
        "Classifier trained."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print()
    print(
        f"Generating predictions "
        f"with threshold={THRESHOLD}"
    )

    predictions = generate_predictions(
        classifier,
        X_test
    )

    print(
        f"Prediction shape: "
        f"{predictions.shape}"
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    print()
    print(
        "Generating confusion matrices..."
    )

    plot_confusion_matrices(
        Y_test,
        predictions
    )

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()