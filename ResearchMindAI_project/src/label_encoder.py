"""
ResearchMindAI — Label Encoding

Paper'ların multi-label kategorilerini
binary label matrix formatına dönüştürür.

Örnek:

["NLP"]
→ [1, 0, 0, 0]

["NLP", "Machine Learning"]
→ [1, 0, 1, 0]
"""

import json
import numpy as np

from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = Path(
    "data/processed/train_papers_final.json"
)

VALIDATION_PATH = Path(
    "data/processed/validation_papers_final.json"
)

TEST_PATH = Path(
    "data/processed/test_papers_final.json"
)

OUTPUT_DIR = Path(
    "data/processed/labels"
)

CLASSES = [
    "NLP",
    "Computer Vision",
    "Machine Learning",
    "Robotics"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_papers(path):
    """
    JSON dosyasını okuyup paper listesini döndürür.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        papers = json.load(file)

    return papers


# ============================================================
# EXTRACT LABELS
# ============================================================

def extract_labels(papers):
    """
    Paper listesinden labels alanını çıkarır.

    Örnek:

    [
        {"labels": ["NLP"]},
        {"labels": ["Computer Vision", "Machine Learning"]}
    ]

    ↓

    [
        ["NLP"],
        ["Computer Vision", "Machine Learning"]
    ]
    """

    labels = []

    for paper in papers:

        labels.append(
            paper["labels"]
        )

    return labels


# ============================================================
# BUILD LABEL MATRIX
# ============================================================

def build_label_matrix(
    train_labels,
    validation_labels,
    test_labels
):
    """
    Multi-label kategorileri binary matrix'e dönüştürür.

    Class sırası:

    [NLP, Computer Vision, Machine Learning, Robotics]

    Örnek:

    ["NLP"]
    → [1, 0, 0, 0]

    ["NLP", "Machine Learning"]
    → [1, 0, 1, 0]
    """

    mlb = MultiLabelBinarizer(
        classes=CLASSES
    )

    # Train üzerinde encoder'ı öğreniyoruz.
    Y_train = mlb.fit_transform(
        train_labels
    )

    # Validation ve Test için
    # aynı class sırasını kullanıyoruz.
    Y_validation = mlb.transform(
        validation_labels
    )

    Y_test = mlb.transform(
        test_labels
    )

    return (
        mlb,
        Y_train,
        Y_validation,
        Y_test
    )


# ============================================================
# SAVE LABEL MATRICES
# ============================================================

def save_labels(
    mlb,
    Y_train,
    Y_validation,
    Y_test
):
    """
    Label matrix'lerini .npy formatında kaydeder.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR / "Y_train_final.npy",
        Y_train
    )

    np.save(
        OUTPUT_DIR / "Y_validation_final.npy",
        Y_validation
    )

    np.save(
        OUTPUT_DIR / "Y_test_final.npy",
        Y_test
    )

    # Class isimlerini de kaydediyoruz.
    np.save(
        OUTPUT_DIR / "classes_final.npy",
        mlb.classes_
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — LABEL ENCODING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_papers = load_papers(
        TRAIN_PATH
    )

    validation_papers = load_papers(
        VALIDATION_PATH
    )

    test_papers = load_papers(
        TEST_PATH
    )

    print(
        f"Train papers: {len(train_papers)}"
    )

    print(
        f"Validation papers: {len(validation_papers)}"
    )

    print(
        f"Test papers: {len(test_papers)}"
    )

    # --------------------------------------------------------
    # Extract labels
    # --------------------------------------------------------

    train_labels = extract_labels(
        train_papers
    )

    validation_labels = extract_labels(
        validation_papers
    )

    test_labels = extract_labels(
        test_papers
    )

    # --------------------------------------------------------
    # Build label matrices
    # --------------------------------------------------------

    (
        mlb,
        Y_train,
        Y_validation,
        Y_test
    ) = build_label_matrix(
        train_labels,
        validation_labels,
        test_labels
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("Classes:")
    print(mlb.classes_)

    print()
    print(
        "Y_train shape:",
        Y_train.shape
    )

    print(
        "Y_validation shape:",
        Y_validation.shape
    )

    print(
        "Y_test shape:",
        Y_test.shape
    )

    print()
    print("First 5 train labels:")

    print(
        Y_train[:5]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_labels(
        mlb,
        Y_train,
        Y_validation,
        Y_test
    )

    print()
    print("=" * 70)
    print("LABEL ENCODING COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
