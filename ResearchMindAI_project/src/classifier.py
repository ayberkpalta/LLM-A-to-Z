"""
ResearchMindAI — Multi-Label Classifier

MPNet embedding'lerini kullanarak
paper kategorilerini tahmin eden
Multi-Label Logistic Regression modeli.

Input:
    X_train → (1271, 768)
    Y_train → (1271, 4)

Output:
    Eğitilmiş classifier
"""

import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_DIR = Path(
    "data/processed/embeddings"
)

LABEL_DIR = Path(
    "data/processed/labels"
)

MODEL_DIR = Path(
    "models"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():
    """
    Daha önce MPNet ile oluşturduğumuz
    embedding'leri yükler.
    """

    X_train = np.load(
        EMBEDDING_DIR / "train_embeddings_final.npy"
    )

    X_validation = np.load(
        EMBEDDING_DIR / "validation_embeddings_final.npy"
    )

    X_test = np.load(
        EMBEDDING_DIR / "test_embeddings_final.npy"
    )

    return (
        X_train,
        X_validation,
        X_test
    )


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():
    """
    Multi-label binary matrix'lerini yükler.
    """

    Y_train = np.load(
        LABEL_DIR / "Y_train_final.npy"
    )

    Y_validation = np.load(
        LABEL_DIR / "Y_validation_final.npy"
    )

    Y_test = np.load(
        LABEL_DIR / "Y_test_final.npy"
    )

    classes = np.load(
        LABEL_DIR / "classes_final.npy",
        allow_pickle=True
    )

    return (
        Y_train,
        Y_validation,
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
    """
    Multi-label Logistic Regression modelini eğitir.

    Her class için ayrı binary classifier oluşturulur.

    Örneğin:

    NLP
    CV
    ML
    Robotics
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — CLASSIFIER TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test
    ) = load_embeddings()

    print()
    print("EMBEDDINGS")
    print("-" * 70)

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_validation:",
        X_validation.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    (
        Y_train,
        Y_validation,
        Y_test,
        classes
    ) = load_labels()

    print()
    print("LABELS")
    print("-" * 70)

    print(
        "Y_train:",
        Y_train.shape
    )

    print(
        "Y_validation:",
        Y_validation.shape
    )

    print(
        "Y_test:",
        Y_test.shape
    )

    print()
    print("Classes:")
    print(classes)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)

    classifier = train_classifier(
        X_train,
        Y_train
    )

    print()
    print("Embedding classifier trained successfully!")

    # --------------------------------------------------------
    # Number of classifiers
    # --------------------------------------------------------

    print()
    print(
        "Number of binary classifiers:",
        len(classifier.estimators_)
    )

    for class_name, estimator in zip(
        classes,
        classifier.estimators_
    ):

        print(
            f"{class_name}: "
            f"Logistic Regression"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
