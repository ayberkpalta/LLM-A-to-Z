# ============================================================
# RESEARCHMINDAI — PAPER PREDICTION
# ============================================================

import os
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/researchmindai_svm_final.joblib"
METADATA_PATH = "models/model_metadata.npz"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Daha önce eğittiğimiz final SVM modelini yükler.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    """
    Model ile birlikte kaydettiğimiz sınıf isimlerini yükler.
    """

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata bulunamadı: {METADATA_PATH}"
        )

    metadata = np.load(
        METADATA_PATH,
        allow_pickle=True
    )

    classes = metadata["classes"]

    return classes


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """
    Paper'ın text'ini 768 boyutlu embedding'e
    dönüştürecek MPNet modelini yükler.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL_NAME}"
    )

    return model


# ============================================================
# GENERATE EMBEDDING
# ============================================================

def generate_embedding(
    embedding_model,
    text
):
    """
    Yeni paper'ın text'ini embedding'e dönüştürür.
    """

    embedding = embedding_model.encode(
        [text],
        show_progress_bar=False
    )

    return embedding


# ============================================================
# PREDICT PAPER
# ============================================================

def predict_paper(
    classifier,
    embedding,
    classes
):
    """
    Embedding üzerinden paper'ın hangi kategorilere
    ait olduğunu tahmin eder.
    """

    prediction = classifier.predict(
        embedding
    )

    prediction = prediction[0]

    predicted_classes = []

    for index, value in enumerate(prediction):

        if value == 1:
            predicted_classes.append(
                classes[index]
            )

    return predicted_classes, prediction


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — PAPER PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD CLASSIFIER
    # --------------------------------------------------------

    print("\nLoading final SVM...")

    classifier = load_model()

    print("Final SVM loaded successfully.")

    # --------------------------------------------------------
    # LOAD CLASSES
    # --------------------------------------------------------

    classes = load_metadata()

    print("\nClasses:")
    print(classes)

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = load_embedding_model()

    # --------------------------------------------------------
    # GET PAPER TEXT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ENTER PAPER")
    print("=" * 70)

    print(
        "\nPaper abstract/text'i gir."
        "\nBitirmek için ENTER'a iki kez basabilirsin.\n"
    )

    text_lines = []

    while True:

        line = input()

        if line.strip() == "":
            break

        text_lines.append(line)

    text = " ".join(text_lines)

    if not text.strip():

        print("\nNo text entered.")

        return

    # --------------------------------------------------------
    # GENERATE EMBEDDING
    # --------------------------------------------------------

    print("\nGenerating embedding...")

    embedding = generate_embedding(
        embedding_model,
        text
    )

    print(
        f"Embedding shape: {embedding.shape}"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nGenerating prediction...")

    predicted_classes, prediction = predict_paper(
        classifier,
        embedding,
        classes
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print("\nBinary prediction:")

    for index, class_name in enumerate(classes):

        print(
            f"{class_name:<20}: {prediction[index]}"
        )

    print("\nPredicted categories:")

    if predicted_classes:

        for class_name in predicted_classes:

            print(
                f"✓ {class_name}"
            )

    else:

        print(
            "No category detected."
        )

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()