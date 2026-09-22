import os
import pickle
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pinecone import Pinecone

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

HF_REPO_ID = "Satyam-0001/readmission-artifacts"
BATCH_SIZE = 100
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "readmission")


def get_metadata(match_value: Any) -> dict[str, int]:
    return {"readmitted": int(match_value)}


def reconstruct_vectors(index: faiss.Index) -> np.ndarray:
    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    for vector_id in range(index.ntotal):
        vectors[vector_id] = index.reconstruct(vector_id)
    return vectors


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        raise RuntimeError(
            "Set PINECONE_API_KEY and PINECONE_INDEX_NAME in Backend/.env first."
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    outcomes_path = os.path.join(base_dir, "artifacts", "train_outcomes.pkl")
    faiss_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="faiss_index.bin",
        repo_type="dataset",
    )

    index = faiss.read_index(faiss_path)
    with open(outcomes_path, "rb") as file:
        train_outcomes = np.asarray(pickle.load(file)).reshape(-1)

    if index.ntotal != len(train_outcomes):
        raise ValueError(
            f"FAISS vectors ({index.ntotal}) and outcomes ({len(train_outcomes)}) "
            "do not have the same length."
        )

    print(f"Loaded {index.ntotal} vectors with dimension {index.d}.")
    vectors = reconstruct_vectors(index)

    pinecone = Pinecone(api_key=api_key)
    pinecone_index = pinecone.Index(index_name)

    for start in range(0, index.ntotal, BATCH_SIZE):
        end = min(start + BATCH_SIZE, index.ntotal)
        batch = [
            {
                "id": f"patient-{vector_id}",
                "values": vectors[vector_id].tolist(),
                "metadata": get_metadata(train_outcomes[vector_id]),
            }
            for vector_id in range(start, end)
        ]
        pinecone_index.upsert(vectors=batch, namespace=NAMESPACE)
        print(f"Uploaded {end}/{index.ntotal} vectors.")

    print("Migration complete.")
    print(f"Index: {index_name}")
    print(f"Namespace: {NAMESPACE}")


if __name__ == "__main__":
    main()
