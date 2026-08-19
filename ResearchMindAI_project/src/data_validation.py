"""
classification_papers_final.json
            ↓
       VALIDATION
            ↓
 ┌──────────────────────┐
 │ Toplam paper         │
 │ Unique ID            │
 │ Duplicate ID         │
 │ Boş title            │
 │ Boş abstract         │
 │ Boş authors          │
 │ Boş categories       │
 │ Boş labels           │
 │ Geçersiz labels      │
 │ Label dağılımı       │
 │ Multi-label dağılımı │
 │ Abstract uzunlukları │
 └──────────────────────┘
            ↓
      DATA QUALITY REPORT
 
 veriyi değiştirmek değil, önce datasetimizin gerçekten sağlıklı olup olmadığını kontrol etmek.
 """

"""
ResearchMindAI
Data Validation Pipeline

Görev:
ArXiv'den toplanan final dataset'in veri kalitesini kontrol etmek.

Input:
    data/raw/classification_papers_final.json

Output:
    Terminal üzerinde DATA VALIDATION REPORT
"""

import json
from collections import Counter
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/raw/classification_papers_final.json"
)

EXPECTED_LABELS = {
    "NLP",
    "Computer Vision",
    "Machine Learning",
    "Robotics"
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(input_path):
    """
    JSON dataset'i yükler.
    """

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        papers = json.load(file)

    return papers


# ============================================================
# BASIC DATASET CHECK
# ============================================================

def validate_basic_structure(papers):
    """
    Dataset'in temel yapısını kontrol eder.
    """

    print()
    print("=" * 70)
    print("BASIC DATASET VALIDATION")
    print("=" * 70)

    print(f"Total papers: {len(papers)}")

    if not isinstance(papers, list):
        print("ERROR: Dataset is not a list.")
        return

    print("Dataset structure: OK")


# ============================================================
# REQUIRED FIELD VALIDATION
# ============================================================

def validate_required_fields(papers):
    """
    Her paper'ın gerekli alanlara sahip olup olmadığını kontrol eder.
    """

    required_fields = [
        "id",
        "title",
        "abstract",
        "authors",
        "categories",
        "published",
        "paper_url",
        "pdf_url",
        "labels"
    ]

    missing_fields = Counter()

    for paper in papers:

        for field in required_fields:

            if field not in paper:
                missing_fields[field] += 1

    print()
    print("=" * 70)
    print("REQUIRED FIELD VALIDATION")
    print("=" * 70)

    if not missing_fields:

        print("All required fields are present.")

    else:

        print("Missing fields:")

        for field, count in missing_fields.items():

            print(
                f"{field}: {count}"
            )


# ============================================================
# EMPTY VALUE VALIDATION
# ============================================================

def validate_empty_values(papers):
    """
    Boş veya None değerleri kontrol eder.
    """

    fields = [
        "id",
        "title",
        "abstract",
        "authors",
        "categories",
        "published",
        "paper_url",
        "pdf_url",
        "labels"
    ]

    empty_counts = Counter()

    for paper in papers:

        for field in fields:

            value = paper.get(field)

            if value is None:

                empty_counts[field] += 1

            elif isinstance(value, str):

                if not value.strip():

                    empty_counts[field] += 1

            elif isinstance(value, list):

                if len(value) == 0:

                    empty_counts[field] += 1

    print()
    print("=" * 70)
    print("EMPTY VALUE VALIDATION")
    print("=" * 70)

    if not empty_counts:

        print("No empty values found.")

    else:

        for field, count in empty_counts.items():

            print(
                f"{field}: {count}"
            )


# ============================================================
# DUPLICATE ID VALIDATION
# ============================================================

def validate_duplicate_ids(papers):
    """
    ID duplicate kontrolü yapar.
    """

    ids = [
        paper.get("id")
        for paper in papers
    ]

    id_counts = Counter(ids)

    duplicates = {
        paper_id: count
        for paper_id, count in id_counts.items()
        if count > 1
    }

    print()
    print("=" * 70)
    print("DUPLICATE ID VALIDATION")
    print("=" * 70)

    print(
        f"Unique IDs: {len(id_counts)}"
    )

    print(
        f"Duplicate IDs: {len(duplicates)}"
    )

    if duplicates:

        print("\nDuplicate examples:")

        for paper_id, count in list(
            duplicates.items()
        )[:10]:

            print(
                f"{paper_id} -> {count}"
            )

    else:

        print("No duplicate IDs found.")


# ============================================================
# DUPLICATE TITLE VALIDATION
# ============================================================

def validate_duplicate_titles(papers):
    """
    Title duplicate kontrolü yapar.
    """

    titles = [
        paper.get("title", "").strip().lower()
        for paper in papers
    ]

    title_counts = Counter(titles)

    duplicates = {
        title: count
        for title, count in title_counts.items()
        if title and count > 1
    }

    print()
    print("=" * 70)
    print("DUPLICATE TITLE VALIDATION")
    print("=" * 70)

    print(
        f"Unique titles: {len(title_counts)}"
    )

    print(
        f"Duplicate titles: {len(duplicates)}"
    )

    if duplicates:

        print("\nDuplicate title examples:")

        for title, count in list(
            duplicates.items()
        )[:10]:

            print(
                f"{title} -> {count}"
            )


# ============================================================
# LABEL VALIDATION
# ============================================================

def validate_labels(papers):
    """
    Label'ların beklenen sınıflardan olup olmadığını kontrol eder.
    """

    label_counts = Counter()
    invalid_labels = set()

    for paper in papers:

        labels = paper.get(
            "labels",
            []
        )

        for label in labels:

            label_counts[label] += 1

            if label not in EXPECTED_LABELS:

                invalid_labels.add(label)

    print()
    print("=" * 70)
    print("LABEL VALIDATION")
    print("=" * 70)

    print("Label distribution:")

    for label in sorted(
        EXPECTED_LABELS
    ):

        print(
            f"{label}: "
            f"{label_counts[label]}"
        )

    if invalid_labels:

        print()
        print("Invalid labels:")

        for label in sorted(
            invalid_labels
        ):

            print(label)

    else:

        print()
        print("No invalid labels found.")


# ============================================================
# MULTI-LABEL VALIDATION
# ============================================================

def validate_multilabel_distribution(papers):
    """
    Paper'ların kaç farklı label'a sahip olduğunu analiz eder.
    """

    label_count_distribution = Counter()

    for paper in papers:

        number_of_labels = len(
            paper.get(
                "labels",
                []
            )
        )

        label_count_distribution[
            number_of_labels
        ] += 1

    print()
    print("=" * 70)
    print("MULTI-LABEL DISTRIBUTION")
    print("=" * 70)

    for number_of_labels in sorted(
        label_count_distribution
    ):

        count = label_count_distribution[
            number_of_labels
        ]

        print(
            f"{number_of_labels} label(s): "
            f"{count} papers"
        )


# ============================================================
# ABSTRACT LENGTH VALIDATION
# ============================================================

def validate_abstract_lengths(papers):
    """
    Abstract uzunluklarının temel istatistiklerini çıkarır.
    """

    lengths = []

    for paper in papers:

        abstract = paper.get(
            "abstract",
            ""
        )

        if isinstance(
            abstract,
            str
        ) and abstract.strip():

            lengths.append(
                len(
                    abstract.split()
                )
            )

    print()
    print("=" * 70)
    print("ABSTRACT LENGTH VALIDATION")
    print("=" * 70)

    if not lengths:

        print("No valid abstracts found.")
        return

    print(
        f"Minimum words: {min(lengths)}"
    )

    print(
        f"Maximum words: {max(lengths)}"
    )

    print(
        f"Average words: "
        f"{sum(lengths) / len(lengths):.2f}"
    )


# ============================================================
# FINAL VALIDATION SUMMARY
# ============================================================

def print_final_summary(papers):
    """
    Genel dataset özetini gösterir.
    """

    unique_ids = len(
        set(
            paper.get("id")
            for paper in papers
        )
    )

    multilabel_papers = sum(
        1
        for paper in papers
        if len(
            paper.get(
                "labels",
                []
            )
        ) > 1
    )

    print()
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Total papers: {len(papers)}"
    )

    print(
        f"Unique IDs: {unique_ids}"
    )

    print(
        f"Multi-label papers: "
        f"{multilabel_papers}"
    )

    print(
        f"Single-label papers: "
        f"{len(papers) - multilabel_papers}"
    )

    print()
    print("Validation completed.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — DATA VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    papers = load_dataset(
        INPUT_PATH
    )

    # --------------------------------------------------------
    # Validation steps
    # --------------------------------------------------------

    validate_basic_structure(
        papers
    )

    validate_required_fields(
        papers
    )

    validate_empty_values(
        papers
    )

    validate_duplicate_ids(
        papers
    )

    validate_duplicate_titles(
        papers
    )

    validate_labels(
        papers
    )

    validate_multilabel_distribution(
        papers
    )

    validate_abstract_lengths(
        papers
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_final_summary(
        papers
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

