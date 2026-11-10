"""Find the most relevant stored ShopRAG chunks for a customer question."""

import math
import sys
from pathlib import Path
from urllib.parse import quote, quote_plus


MODEL_NAME = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 3072
COLLECTION_NAME = "shop_products"
RESULT_COUNT = 3


def safe_error_message(error, api_key):
    """Format an error without accidentally showing the API key."""
    message = str(error) or "No additional details were provided."

    # Redact the key in case an SDK or HTTP error includes it in its message.
    for secret_form in (api_key, quote(api_key, safe=""), quote_plus(api_key)):
        if secret_form:
            message = message.replace(secret_form, "[REDACTED]")

    return f"{type(error).__name__}: {message}"


def main():
    # Point to the existing local database without creating a new one.
    project_folder = Path(__file__).resolve().parent
    database_path = project_folder / "chroma_db"
    sqlite_path = database_path / "chroma.sqlite3"

    if not database_path.is_dir() or not sqlite_path.is_file():
        print(f"Error: Existing ChromaDB database was not found at {database_path}")
        return 1

    try:
        import chromadb

        # Get the existing collection. This does not create or modify it.
        chroma_client = chromadb.PersistentClient(path=str(database_path))
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except ImportError:
        print("Error: ChromaDB is not installed. Install it with: pip install chromadb")
        return 1
    except Exception as error:
        print(f"Error opening the ChromaDB collection: {type(error).__name__}: {error}")
        return 1

    # Ask for one customer question. Strip surrounding whitespace before using it.
    try:
        question = input("Enter your question: ").strip()
    except EOFError:
        print("Error: No question was provided.")
        return 1
    if not question:
        print("Error: The question cannot be empty.")
        return 1

    if collection.count() == 0:
        print(f"Error: The {COLLECTION_NAME} collection contains no records.")
        return 1

    try:
        # config.py reads and validates GEMINI_API_KEY from the existing .env.
        from config import GEMINI_API_KEY
    except Exception as error:
        print(f"Error loading the Gemini API key: {error}")
        return 1

    try:
        # Use the official Google GenAI SDK with the same model and vector
        # dimensionality used to create the stored document embeddings.
        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        response = gemini_client.models.embed_content(
            model=MODEL_NAME,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            ),
        )

        if not response.embeddings:
            raise ValueError("The API response did not contain an embedding.")

        embedding_values = response.embeddings[0].values
        if not embedding_values:
            raise ValueError("The API response contained an empty embedding vector.")

        question_embedding = [float(value) for value in embedding_values]
        if len(question_embedding) != OUTPUT_DIMENSIONALITY or not all(
            math.isfinite(value) for value in question_embedding
        ):
            raise ValueError("The API returned an invalid embedding vector.")
    except ImportError:
        print("Error: Google GenAI SDK is not installed. Install it with: pip install google-genai")
        return 1
    except Exception as error:
        print(
            "Error creating the question embedding: "
            f"{safe_error_message(error, GEMINI_API_KEY)}"
        )
        return 1

    try:
        # Query ChromaDB with the question vector. This is read-only and does
        # not regenerate or update the stored document embeddings.
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=RESULT_COUNT,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as error:
        print(f"Error searching ChromaDB: {type(error).__name__}: {error}")
        return 1

    result_ids = results.get("ids", [[]])[0]
    result_documents = results.get("documents", [[]])[0]
    result_metadatas = results.get("metadatas", [[]])[0]
    result_distances = results.get("distances", [[]])[0]

    if not result_ids:
        print("No matching chunks were found.")
        return 0

    print(f"Top {len(result_ids)} relevant chunks:")
    for number, chunk_id in enumerate(result_ids, start=1):
        document = result_documents[number - 1] if result_documents else ""
        metadata = result_metadatas[number - 1] if result_metadatas else {}
        distance = result_distances[number - 1] if result_distances else None

        # Use the stored metadata ID when available, falling back to the
        # collection's record ID.
        displayed_id = metadata.get("chunk_id", chunk_id) if metadata else chunk_id
        if distance is None:
            distance_text = "unavailable"
        else:
            distance_text = f"{distance:.6f}"

        print(f"\nResult {number}")
        print(f"Chunk ID: {displayed_id}")
        print(f"Distance: {distance_text}")
        print("Original text:")
        print(document)

    return 0


if __name__ == "__main__":
    sys.exit(main())
