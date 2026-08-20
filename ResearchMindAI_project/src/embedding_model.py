"""
ResearchMindAI — Embedding Generation

Train / Validation / Test datasetlerinden
sentence embedding üretir ve kaydeder.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


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
    "data/processed/embeddings"
)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


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
# EXTRACT TEXT
# ============================================================

def extract_texts(papers):
    """
    Paper'ların text alanlarını çıkarır.
    """

    texts = []

    for paper in papers:

        texts.append(
            paper["text"]
        )

    return texts


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(
    model,
    texts
):
    """
    Verilen text'ler için embedding üretir.
    """

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return np.array(embeddings)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    embeddings,
    output_path
):
    """
    Embedding array'ini .npy olarak kaydeder.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_path,
        embeddings
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Shape: {embeddings.shape}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — EMBEDDING GENERATION")
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
    # Extract texts
    # --------------------------------------------------------

    train_texts = extract_texts(
        train_papers
    )

    validation_texts = extract_texts(
        validation_papers
    )

    test_texts = extract_texts(
        test_papers
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print()
    print("Generating train embeddings...")

    train_embeddings = generate_embeddings(
        model,
        train_texts
    )

    print()
    print("Generating validation embeddings...")

    validation_embeddings = generate_embeddings(
        model,
        validation_texts
    )

    print()
    print("Generating test embeddings...")

    test_embeddings = generate_embeddings(
        model,
        test_texts
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_embeddings(
        train_embeddings,
        OUTPUT_DIR / "train_embeddings_final.npy"
    )

    save_embeddings(
        validation_embeddings,
        OUTPUT_DIR / "validation_embeddings_final.npy"
    )

    save_embeddings(
        test_embeddings,
        OUTPUT_DIR / "test_embeddings_final.npy"
    )

    print()
    print("=" * 70)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()