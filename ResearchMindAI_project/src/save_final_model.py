
# ============================================================
# RESEARCHMINDAI
# FINAL MODEL SAVING
# MPNet Embeddings + One-vs-Rest Linear SVM
# ============================================================

import os
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import joblib


# ============================================================
# PATHS
# ============================================================

TRAIN_EMBEDDINGS_PATH = (
    "data/processed/embeddings/train_embeddings_final.npy"
)

TRAIN_LABELS_PATH = (
    "data/processed/labels/Y_train_final.npy"
)

CLASSES_PATH = (
    "data/processed/labels/classes_final.npy"
)

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "researchmindai_svm_final.joblib"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.npz"
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

BEST_C = 0.25


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    X_train = np.load(
        TRAIN_EMBEDDINGS_PATH
    )

    Y_train = np.load(
        TRAIN_LABELS_PATH
    )

    classes = np.load(
        CLASSES_PATH,
        allow_pickle=True
    )

    return (
        X_train,
        Y_train,
        classes
    )


# ============================================================
# TRAIN FINAL SVM
# ============================================================

def train_final_svm(
    X_train,
    Y_train
):

    print("=" * 70)
    print("TRAINING FINAL SVM")
    print("=" * 70)

    print(f"C = {BEST_C}")

    # --------------------------------------------------------
    # Base SVM
    # --------------------------------------------------------

    base_svm = LinearSVC(
        C=BEST_C,
        random_state=42,
        max_iter=10000
    )

    # --------------------------------------------------------
    # One-vs-Rest
    #
    # Her sınıf için ayrı binary classifier oluşturur.
    #
    # NLP:
    #     NLP vs Not NLP
    #
    # Computer Vision:
    #     CV vs Not CV
    #
    # Machine Learning:
    #     ML vs Not ML
    #
    # Robotics:
    #     Robotics vs Not Robotics
    # --------------------------------------------------------

    model = OneVsRestClassifier(
        base_svm
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        Y_train
    )

    print()
    print("Final SVM trained successfully.")

    print(
        f"Number of binary classifiers: "
        f"{len(model.estimators_)}"
    )

    return model


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    classes
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    np.savez(
        METADATA_PATH,
        classes=classes,
        C=BEST_C,
        embedding_model="sentence-transformers/all-mpnet-base-v2"
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Model:    {MODEL_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RESEARCHMINDAI — SAVE FINAL MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, Y_train, classes = (
        load_training_data()
    )

    print()
    print("TRAINING DATA")
    print("-" * 70)

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"Y_train: {Y_train.shape}"
    )

    print(
        f"Classes: {classes}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_final_svm(
        X_train,
        Y_train
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        model,
        classes
    )

    print()
    print("=" * 70)
    print("FINAL MODEL SAVING COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
