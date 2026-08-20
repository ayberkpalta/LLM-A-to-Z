"""
ResearchMindAI — Error Analysis

Amaç:
Eğitilmiş Multi-Label Logistic Regression modelinin
test setindeki hatalarını incelemek.

İncelenen hatalar:
- True Positive
- False Positive
- False Negative
- True Negative

Özellikle Machine Learning sınıfındaki
False Positive ve False Negative örnekleri incelenir.
"""

import json
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

TEST_DATA_PATH = Path(
    "data/processed/test_papers_final.json"
)

THRESHOLD = 0.45

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
    """
    Train/Test embeddinglerini,
    train/test label'larını ve
    test paper'larını yükler.
    """

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

    with open(
        TEST_DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        test_papers = json.load(file)

    return (
        X_train,
        X_test,
        Y_train,
        Y_test,
        test_papers
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

    OneVsRestClassifier:
    Her kategori için ayrı bir binary classifier oluşturur.

    4 kategori olduğu için:

    NLP → Logistic Regression
    Computer Vision → Logistic Regression
    Machine Learning → Logistic Regression
    Robotics → Logistic Regression
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
# GENERATE TEST PREDICTIONS
# ============================================================

def generate_predictions(
    classifier,
    X_test
):
    """
    Test verileri için:

    1. Probability
    2. Binary prediction

    üretir.
    """

    probabilities = classifier.predict_proba(
        X_test
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return (
        probabilities,
        predictions
    )


# ============================================================
# CONVERT LABEL VECTOR TO LABEL NAMES
# ============================================================

def get_label_names(
    label_vector
):
    """
    Binary label vector'ı gerçek
    kategori isimlerine dönüştürür.

    Örneğin:

    [1, 0, 1, 0]

    →

    ['NLP', 'Machine Learning']
    """

    labels = []

    for index, value in enumerate(
        label_vector
    ):

        if value == 1:

            labels.append(
                CLASSES[index]
            )

    return labels


# ============================================================
# CLASS ERROR ANALYSIS
# ============================================================

def analyze_class_errors(
    Y_test,
    predictions
):
    """
    Her sınıf için:

    TP
    FP
    FN
    TN

    değerlerini hesaplar.
    """

    print()
    print("=" * 70)
    print("CLASS ERROR ANALYSIS")
    print("=" * 70)

    for index, class_name in enumerate(
        CLASSES
    ):

        true_labels = Y_test[:, index]

        predicted_labels = predictions[:, index]

        # ----------------------------------------------------
        # TRUE POSITIVE
        # ----------------------------------------------------

        true_positive = np.sum(
            (true_labels == 1) &
            (predicted_labels == 1)
        )

        # ----------------------------------------------------
        # FALSE POSITIVE
        # ----------------------------------------------------

        false_positive = np.sum(
            (true_labels == 0) &
            (predicted_labels == 1)
        )

        # ----------------------------------------------------
        # FALSE NEGATIVE
        # ----------------------------------------------------

        false_negative = np.sum(
            (true_labels == 1) &
            (predicted_labels == 0)
        )

        # ----------------------------------------------------
        # TRUE NEGATIVE
        # ----------------------------------------------------

        true_negative = np.sum(
            (true_labels == 0) &
            (predicted_labels == 0)
        )

        print()
        print(class_name)
        print("-" * 40)

        print(
            f"True Positive:  {true_positive}"
        )

        print(
            f"False Positive: {false_positive}"
        )

        print(
            f"False Negative: {false_negative}"
        )

        print(
            f"True Negative:  {true_negative}"
        )


# ============================================================
# MACHINE LEARNING ERROR ANALYSIS
# ============================================================

def analyze_machine_learning_errors(
    test_papers,
    Y_test,
    predictions,
    probabilities
):
    """
    Machine Learning sınıfındaki:

    False Negative
    False Positive

    örneklerini gösterir.

    False Negative:
    Gerçekte ML fakat model ML demedi.

    False Positive:
    Gerçekte ML değil fakat model ML dedi.
    """

    ml_index = CLASSES.index(
        "Machine Learning"
    )

    print()
    print("=" * 70)
    print("MACHINE LEARNING ERROR ANALYSIS")
    print("=" * 70)

    false_negatives = []
    false_positives = []

    # ========================================================
    # FIND ERRORS
    # ========================================================

    for index, paper in enumerate(
        test_papers
    ):

        true_value = Y_test[
            index,
            ml_index
        ]

        predicted_value = predictions[
            index,
            ml_index
        ]

        probability = probabilities[
            index,
            ml_index
        ]

        # ----------------------------------------------------
        # FALSE NEGATIVE
        # ----------------------------------------------------

        if (
            true_value == 1
            and predicted_value == 0
        ):

            false_negatives.append(
                (
                    index,
                    paper,
                    probability
                )
            )

        # ----------------------------------------------------
        # FALSE POSITIVE
        # ----------------------------------------------------

        if (
            true_value == 0
            and predicted_value == 1
        ):

            false_positives.append(
                (
                    index,
                    paper,
                    probability
                )
            )

    # ========================================================
    # FALSE NEGATIVES
    # ========================================================

    print()
    print(
        f"Machine Learning False Negatives: "
        f"{len(false_negatives)}"
    )

    print()

    for (
        index,
        paper,
        probability
    ) in false_negatives[:10]:

        print("-" * 70)

        print(
            f"Test index: {index}"
        )

        print(
            f"Probability: "
            f"{probability:.4f}"
        )

        print(
            f"True labels: "
            f"{paper['labels']}"
        )

        print(
            f"Text: "
            f"{paper['text'][:500]}..."
        )

    # ========================================================
    # FALSE POSITIVES
    # ========================================================

    print()
    print(
        f"Machine Learning False Positives: "
        f"{len(false_positives)}"
    )

    print()

    for (
        index,
        paper,
        probability
    ) in false_positives[:10]:

        print("-" * 70)

        print(
            f"Test index: {index}"
        )

        print(
            f"Probability: "
            f"{probability:.4f}"
        )

        print(
            f"True labels: "
            f"{paper['labels']}"
        )

        print(
            f"Text: "
            f"{paper['text'][:500]}..."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — ERROR ANALYSIS")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        test_papers
    ) = load_data()

    print()
    print(
        f"Train papers: "
        f"{len(X_train)}"
    )

    print(
        f"Test papers: "
        f"{len(X_test)}"
    )

    print(
        f"Test JSON papers: "
        f"{len(test_papers)}"
    )

    # ========================================================
    # TRAIN CLASSIFIER
    # ========================================================

    print()
    print(
        "Training classifier..."
    )

    classifier = train_classifier(
        X_train,
        Y_train
    )

    print(
        "Classifier trained."
    )

    # ========================================================
    # GENERATE PREDICTIONS
    # ========================================================

    print()
    print(
        f"Generating predictions "
        f"with threshold={THRESHOLD}"
    )

    (
        probabilities,
        predictions
    ) = generate_predictions(
        classifier,
        X_test
    )

    print()

    print(
        f"Probability shape: "
        f"{probabilities.shape}"
    )

    print(
        f"Prediction shape: "
        f"{predictions.shape}"
    )

    # ========================================================
    # CLASS ERROR ANALYSIS
    # ========================================================

    analyze_class_errors(
        Y_test,
        predictions
    )

    # ========================================================
    # MACHINE LEARNING ANALYSIS
    # ========================================================

    analyze_machine_learning_errors(
        test_papers,
        Y_test,
        predictions,
        probabilities
    )

    # ========================================================
    # COMPLETED
    # ========================================================

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
