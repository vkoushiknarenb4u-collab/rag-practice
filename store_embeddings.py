"""Store precomputed ShopRAG vectors in a local ChromaDB collection."""

import json
import math
import sys
from pathlib import Path


COLLECTION_NAME = "shop_products"


def read_embedding_records(path):
    """Read and validate the records created by generate_embeddings.py."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list) or not data:
        raise ValueError("embeddings.json must contain a non-empty JSON list.")

    records = []
    seen_ids = set()
    vector_dimensions = None

    for number, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Embedding record {number} must be a JSON object.")

        chunk_id = item.get("chunk_id")
        chunk_text = item.get("chunk_text")
        embedding = item.get("embedding")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Embedding record {number} has no valid chunk_id.")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID found in embeddings.json: {chunk_id}")
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError(f"Chunk {chunk_id} has empty or invalid text.")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Chunk {chunk_id} has no valid embedding vector.")

        # ChromaDB requires all vectors in a collection to have the same size.
        if vector_dimensions is None:
            vector_dimensions = len(embedding)
        elif len(embedding) != vector_dimensions:
            raise ValueError(
                f"Chunk {chunk_id} has {len(embedding)} values; "
                f"expected {vector_dimensions}."
            )

        # Ensure every vector contains finite numbers before sending it to ChromaDB.
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in embedding
        ):
            raise ValueError(f"Chunk {chunk_id} contains an invalid vector value.")

        seen_ids.add(chunk_id)
        records.append(
            {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "embedding": [float(value) for value in embedding],
            }
        )

    return records


def main():
    # Keep the input file and persistent database in the project folder.
    project_folder = Path(__file__).resolve().parent
    embeddings_path = project_folder / "embeddings.json"
    database_path = project_folder / "chroma_db"

    try:
        records = read_embedding_records(embeddings_path)
    except FileNotFoundError:
        print(f"Error: embeddings.json was not found at {embeddings_path}")
        return 1
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Error reading embeddings.json: {error}")
        return 1

    try:
        import chromadb
    except ImportError:
        print("Error: ChromaDB is not installed. Install it with: pip install chromadb")
        return 1

    try:
        # PersistentClient stores the database files in the chroma_db folder.
        client = chromadb.PersistentClient(path=str(database_path))

        # Reuse the collection if it already exists.
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

        ids = [record["chunk_id"] for record in records]
        documents = [record["chunk_text"] for record in records]
        embeddings = [record["embedding"] for record in records]
        metadatas = [{"chunk_id": record["chunk_id"]} for record in records]

        # Passing the existing embeddings prevents ChromaDB from generating
        # new vectors. Upsert creates missing IDs and updates existing IDs,
        # so rerunning this script does not create duplicate records.
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"Success! Total records stored: {collection.count()}")
        return 0
    except Exception as error:
        print(f"Error storing embeddings in ChromaDB: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
