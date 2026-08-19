"""ArXiv → paperları çek → JSON olarak kaydet"""
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ARXIV_API_URL = "http://export.arxiv.org/api/query"

BATCH_SIZE = 100
PAPERS_PER_CATEGORY = 500

OUTPUT_PATH = Path(
    "data/raw/classification_papers_final.json"
)

NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom"
}


# ArXiv category → Project label
CATEGORY_CONFIG = {
    "cs.CL": "NLP",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.RO": "Robotics"
}


# ============================================================
# FETCH PAPERS FROM ARXIV
# ============================================================

def fetch_arxiv_batch(category, start, batch_size):
    """
    ArXiv API'den belirli bir kategori için
    batch halinde paper çeker.
    """

    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": batch_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    response = requests.get(
        ARXIV_API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    root = ET.fromstring(response.text)

    entries = root.findall(
        "atom:entry",
        NAMESPACE
    )

    return entries


# ============================================================
# PARSE PAPER
# ============================================================

def parse_paper(entry, label):
    """
    ArXiv XML entry'sini Python dictionary'sine dönüştürür.
    """

    paper_id = entry.find(
        "atom:id",
        NAMESPACE
    )

    title = entry.find(
        "atom:title",
        NAMESPACE
    )

    abstract = entry.find(
        "atom:summary",
        NAMESPACE
    )

    published = entry.find(
        "atom:published",
        NAMESPACE
    )

    # --------------------------------------------------------
    # Authors
    # --------------------------------------------------------

    authors = entry.findall(
        "atom:author",
        NAMESPACE
    )

    author_names = []

    for author in authors:

        name = author.find(
            "atom:name",
            NAMESPACE
        )

        if name is not None:
            author_names.append(
                name.text.strip()
            )

    # --------------------------------------------------------
    # ArXiv Categories
    # --------------------------------------------------------

    categories = entry.findall(
        "atom:category",
        NAMESPACE
    )

    category_names = []

    for category_element in categories:

        term = category_element.get("term")

        if term:
            category_names.append(term)

    # --------------------------------------------------------
    # Paper URLs
    # --------------------------------------------------------

    links = entry.findall(
        "atom:link",
        NAMESPACE
    )

    paper_url = None
    pdf_url = None

    for link in links:

        link_type = link.get("type")
        link_href = link.get("href")

        if link_type == "text/html":
            paper_url = link_href

        elif link_type == "application/pdf":
            pdf_url = link_href

    # --------------------------------------------------------
    # Final paper object
    # --------------------------------------------------------

    paper = {
        "id": paper_id.text.strip(),

        "title": " ".join(
            title.text.split()
        ),

        "abstract": " ".join(
            abstract.text.split()
        ),

        "authors": author_names,

        "categories": category_names,

        "published": published.text.strip(),

        "paper_url": paper_url,

        "pdf_url": pdf_url,

        "labels": [label]
    }

    return paper


# ============================================================
# COLLECT ONE CATEGORY
# ============================================================

def collect_category(category, label, target_count):
    """
    Belirli bir ArXiv kategorisinden target_count
    kadar paper toplar.
    """

    collected = []

    start = 0

    print()
    print("=" * 70)
    print(f"COLLECTING: {label}")
    print("=" * 70)

    while len(collected) < target_count:

        print(
            f"Fetching papers "
            f"{start} - {start + BATCH_SIZE}"
        )

        entries = fetch_arxiv_batch(
            category=category,
            start=start,
            batch_size=BATCH_SIZE
        )

        if not entries:

            print(
                "No more papers returned."
            )

            break

        for entry in entries:

            paper = parse_paper(
                entry=entry,
                label=label
            )

            collected.append(paper)

            if len(collected) >= target_count:
                break

        start += BATCH_SIZE

        print(
            f"Collected "
            f"{len(collected)}/{target_count}"
        )

        # ArXiv API'ye aşırı yük bindirmemek için
        time.sleep(3)

    return collected


# ============================================================
# MERGE DUPLICATES
# ============================================================

def merge_duplicate_papers(papers):
    """
    Aynı paper farklı kategorilerde bulunuyorsa
    ID üzerinden tek kayıtta birleştirir.

    Örnek:

    Paper A → NLP
    Paper A → Machine Learning

    Sonuç:

    Paper A → [NLP, Machine Learning]
    """

    paper_dict = {}

    for paper in papers:

        paper_id = paper["id"]

        # İlk kez görülüyorsa
        if paper_id not in paper_dict:

            paper_dict[paper_id] = paper.copy()

        # Daha önce görülmüşse
        else:

            existing_labels = (
                paper_dict[paper_id]["labels"]
            )

            new_labels = paper["labels"]

            for label in new_labels:

                if label not in existing_labels:

                    existing_labels.append(label)

    return list(
        paper_dict.values()
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(papers, output_path):
    """
    Dataset'i JSON formatında kaydeder.
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
# MAIN PIPELINE
# ============================================================

def main():

    all_papers = []

    # --------------------------------------------------------
    # Collect all categories
    # --------------------------------------------------------

    for category, label in CATEGORY_CONFIG.items():

        papers = collect_category(
            category=category,
            label=label,
            target_count=PAPERS_PER_CATEGORY
        )

        all_papers.extend(papers)

    # --------------------------------------------------------
    # Before duplicate merging
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BEFORE DUPLICATE MERGING")
    print("=" * 70)

    print(
        f"Total papers: {len(all_papers)}"
    )

    # --------------------------------------------------------
    # Merge duplicate papers
    # --------------------------------------------------------

    final_papers = merge_duplicate_papers(
        all_papers
    )

    # --------------------------------------------------------
    # After duplicate merging
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AFTER DUPLICATE MERGING")
    print("=" * 70)

    print(
        f"Unique papers: {len(final_papers)}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        papers=final_papers,
        output_path=OUTPUT_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()