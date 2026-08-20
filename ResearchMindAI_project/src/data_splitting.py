
"""
ResearchMindAI — Dataset Splitting

Processed dataset'i:
    Train
    Validation
    Test

olarak böler.

Multi-label yapıyı korumak için
iterative stratification kullanılır.
"""

import json
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/processed/papers_processed_final.json"
)

TRAIN_OUTPUT_PATH = Path(
    "data/processed/train_papers_final.json"
)

VALIDATION_OUTPUT_PATH = Path(
    "data/processed/validation_papers_final.json"
)

TEST_OUTPUT_PATH = Path(
    "data/processed/test_papers_final.json"
)

RANDOM_STATE = 42

TEST_SIZE = 0.15

VALIDATION_SIZE = 0.15


CLASSES = [
    "NLP",
    "Computer Vision",
    "Machine Learning",
    "Robotics"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(input_path):
    """
    Processed JSON dataset'ini yükler.
    """

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        papers = json.load(file)

    return papers


# ============================================================
# BUILD LABEL MATRIX
# ============================================================

def build_label_matrix(papers):
    """
    Multi-label paper'ları binary label matrix'e dönüştürür.

    Örnek:

    ["NLP"]
        -> [1, 0, 0, 0]

    ["NLP", "Machine Learning"]
        -> [1, 0, 1, 0]
    """

    label_matrix = []

    for paper in papers:

        labels = paper["labels"]

        row = []

        for class_name in CLASSES:

            if class_name in labels:
                row.append(1)
            else:
                row.append(0)

        label_matrix.append(row)

    return np.array(label_matrix)


# ============================================================
# SPLIT TRAIN / TEMP
# ============================================================

def split_train_temp(
    papers,
    labels
):
    """
    Dataset'i:

    Train
    Temp

    olarak böler.

    Temp daha sonra Validation ve Test
    olarak ikiye ayrılır.
    """

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, temp_indices = next(
        splitter.split(
            np.zeros(len(papers)),
            labels
        )
    )

    train_papers = [
        papers[index]
        for index in train_indices
    ]

    temp_papers = [
        papers[index]
        for index in temp_indices
    ]

    temp_labels = labels[temp_indices]

    return (
        train_papers,
        temp_papers,
        temp_labels
    )


# ============================================================
# SPLIT VALIDATION / TEST
# ============================================================

def split_validation_test(
    temp_papers,
    temp_labels
):
    """
    Temp dataset'i Validation ve Test olarak böler.
    """

    # Temp toplamda %30.
    #
    # Validation = %15
    # Test       = %15
    #
    # Dolayısıyla temp'in yarısı validation,
    # yarısı test olacak.

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=RANDOM_STATE
    )

    validation_indices, test_indices = next(
        splitter.split(
            np.zeros(len(temp_papers)),
            temp_labels
        )
    )

    validation_papers = [
        temp_papers[index]
        for index in validation_indices
    ]

    test_papers = [
        temp_papers[index]
        for index in test_indices
    ]

    return (
        validation_papers,
        test_papers
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    papers,
    output_path
):
    """
    Paper listesini JSON olarak kaydeder.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            papers,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def print_label_distribution(
    name,
    papers
):
    """
    Dataset içerisindeki label dağılımını gösterir.
    """

    counts = {
        class_name: 0
        for class_name in CLASSES
    }

    for paper in papers:

        for label in paper["labels"]:

            counts[label] += 1

    print()
    print(
        f"{name} LABEL DISTRIBUTION"
    )

    print("-" * 50)

    for class_name in CLASSES:

        print(
            f"{class_name}: "
            f"{counts[class_name]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — DATA SPLITTING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    papers = load_dataset(
        INPUT_PATH
    )

    print(
        f"Total papers: {len(papers)}"
    )

    # --------------------------------------------------------
    # Build labels
    # --------------------------------------------------------

    labels = build_label_matrix(
        papers
    )

    print(
        f"Label matrix shape: {labels.shape}"
    )

    # --------------------------------------------------------
    # Train / Temp
    # --------------------------------------------------------

    (
        train_papers,
        temp_papers,
        temp_labels
    ) = split_train_temp(
        papers,
        labels
    )

    # --------------------------------------------------------
    # Validation / Test
    # --------------------------------------------------------

    (
        validation_papers,
        test_papers
    ) = split_validation_test(
        temp_papers,
        temp_labels
    )

    # --------------------------------------------------------
    # Print sizes
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT RESULTS")
    print("=" * 70)

    print(
        f"Train:       {len(train_papers)}"
    )

    print(
        f"Validation:  {len(validation_papers)}"
    )

    print(
        f"Test:        {len(test_papers)}"
    )

    print(
        f"Total:       "
        f"{len(train_papers) + len(validation_papers) + len(test_papers)}"
    )

    # --------------------------------------------------------
    # Label distributions
    # --------------------------------------------------------

    print_label_distribution(
        "TRAIN",
        train_papers
    )

    print_label_distribution(
        "VALIDATION",
        validation_papers
    )

    print_label_distribution(
        "TEST",
        test_papers
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        train_papers,
        TRAIN_OUTPUT_PATH
    )

    save_dataset(
        validation_papers,
        VALIDATION_OUTPUT_PATH
    )

    save_dataset(
        test_papers,
        TEST_OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 70)

    print(
        f"Train saved:      {TRAIN_OUTPUT_PATH}"
    )

    print(
        f"Validation saved: {VALIDATION_OUTPUT_PATH}"
    )

    print(
        f"Test saved:       {TEST_OUTPUT_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
