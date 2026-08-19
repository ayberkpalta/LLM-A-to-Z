
"""
ResearchMindAI — Dataset Builder

Final JSON dataset'ini modelleme için uygun
text + labels formatına dönüştürür.
"""

import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/raw/classification_papers_final.json"
)

OUTPUT_PATH = Path(
    "data/processed/papers_processed_final.json"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(input_path):
    """
    JSON dataset'ini yükler.
    """

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        papers = json.load(file)

    return papers


# ============================================================
# BUILD MODEL TEXT
# ============================================================

def build_text(paper):
    """
    Paper'ın title ve abstract alanlarını
    modelin kullanacağı tek bir text alanında birleştirir.
    """

    title = paper["title"].strip()
    abstract = paper["abstract"].strip()

    text = f"{title}. {abstract}"

    return text


# ============================================================
# BUILD PROCESSED DATASET
# ============================================================

def build_dataset(papers):
    """
    Her paper için modelleme aşamasında ihtiyaç
    duyacağımız alanları oluşturur.
    """

    processed_papers = []

    for paper in papers:

        processed_paper = {
            "id": paper["id"],
            "text": build_text(paper),
            "labels": paper["labels"]
        }

        processed_papers.append(
            processed_paper
        )

    return processed_papers


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    papers,
    output_path
):
    """
    İşlenmiş dataset'i JSON olarak kaydeder.
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

    print()
    print("=" * 70)
    print("DATASET SAVED")
    print("=" * 70)

    print(
        f"Path: {output_path}"
    )

    print(
        f"Total papers: {len(papers)}"
    )


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_dataset(papers):
    """
    Oluşturulan dataset'in temel yapısını kontrol eder.
    """

    print()
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print(
        f"Total papers: {len(papers)}"
    )

    required_fields = [
        "id",
        "text",
        "labels"
    ]

    for paper in papers:

        for field in required_fields:

            if field not in paper:

                raise ValueError(
                    f"Missing field: {field}"
                )

    print(
        "Required fields: OK"
    )

    empty_texts = [
        paper
        for paper in papers
        if not paper["text"].strip()
    ]

    if empty_texts:

        raise ValueError(
            f"Empty texts found: {len(empty_texts)}"
        )

    print(
        "Empty texts: 0"
    )

    print(
        "Dataset structure: OK"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — DATASET BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    papers = load_dataset(
        INPUT_PATH
    )

    print(
        f"Loaded papers: {len(papers)}"
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    processed_papers = build_dataset(
        papers
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        processed_papers
    )

    # --------------------------------------------------------
    # Example
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXAMPLE PAPER")
    print("=" * 70)

    print(
        processed_papers[0]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        papers=processed_papers,
        output_path=OUTPUT_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

