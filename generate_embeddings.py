"""Generate and save Gemini embeddings for the ShopRAG text chunks."""

import json
import math
import sys
from pathlib import Path
from urllib.parse import quote, quote_plus


MODEL_NAME = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 3072


def read_chunks(path):
    """Read chunks.json and check that each chunk has an ID and text."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("chunks.json must contain a JSON list of chunks.")

    chunks = []
    seen_ids = set()

    for number, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk record {number} must be a JSON object.")

        chunk_id = item.get("chunk_id")
        chunk_text = item.get("text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chunk record {number} has no valid chunk_id.")
        if not isinstance(chunk_text, str):
            raise ValueError(f"Chunk {chunk_id} has no valid text value.")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID found: {chunk_id}")

        seen_ids.add(chunk_id)
        chunks.append({"chunk_id": chunk_id, "text": chunk_text})

    return chunks


def is_valid_vector(vector):
    """Return True when a saved vector has the expected numeric values."""
    if not isinstance(vector, list) or len(vector) != OUTPUT_DIMENSIONALITY:
        return False

    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    )


def load_saved_embeddings(path, chunks):
    """Load reusable embeddings that still match the source chunk text."""
    if not path.exists():
        return {}

    try:
        saved_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "embeddings.json is not valid JSON. Fix or move it before continuing."
        ) from error

    if not isinstance(saved_data, list):
        raise ValueError("embeddings.json must contain a JSON list of records.")

    current_text_by_id = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
    saved_embeddings = {}

    for record in saved_data:
        if not isinstance(record, dict):
            print("Warning: Ignoring a malformed saved embedding record.")
            continue

        chunk_id = record.get("chunk_id")
        chunk_text = record.get("chunk_text")
        vector = record.get("embedding")

        # Ignore saved records for chunks that are no longer in chunks.json.
        if chunk_id not in current_text_by_id:
            continue

        if chunk_text != current_text_by_id[chunk_id]:
            print(f"Warning: Saved text changed for {chunk_id}; it will be regenerated.")
            continue

        if not isinstance(chunk_text, str) or not chunk_text.strip():
            continue

        if not is_valid_vector(vector):
            print(f"Warning: Saved embedding for {chunk_id} is invalid; it will be regenerated.")
            continue

        saved_embeddings[chunk_id] = {
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "embedding": vector,
        }

    return saved_embeddings


def save_embeddings(path, chunks, embeddings_by_id):
    """Write completed embeddings in the same order as chunks.json."""
    records = [
        embeddings_by_id[chunk["chunk_id"]]
        for chunk in chunks
        if chunk["chunk_id"] in embeddings_by_id
    ]

    # Write to a temporary file first so an interruption cannot leave a
    # partially written embeddings.json file.
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def safe_error_message(error, api_key):
    """Format an error without accidentally displaying the API key."""
    message = str(error) or "No additional details were provided."

    # Redact the key in case an SDK or HTTP error includes it in its message.
    for secret_form in (api_key, quote(api_key, safe=""), quote_plus(api_key)):
        if secret_form:
            message = message.replace(secret_form, "[REDACTED]")

    return f"{type(error).__name__}: {message}"


def print_summary(total, processed, new_embeddings, saved_embeddings, status):
    """Print progress without showing any credentials."""
    dimensions = sorted(
        {
            len(record["embedding"])
            for record in saved_embeddings.values()
            if isinstance(record.get("embedding"), list)
        }
    )

    if len(dimensions) == 1:
        dimension_text = str(dimensions[0])
    elif dimensions:
        dimension_text = "inconsistent: " + ", ".join(map(str, dimensions))
    else:
        dimension_text = "not available"

    print(f"Total chunks processed: {processed} of {total}")
    print(f"Total embeddings generated: {len(saved_embeddings)}")
    print(f"New embeddings generated this run: {new_embeddings}")
    print(f"Embedding vector dimensions: {dimension_text}")
    print(f"Status: {status}")


def main():
    # Keep all input and output paths relative to this script's project folder.
    project_folder = Path(__file__).resolve().parent
    chunks_path = project_folder / "chunks.json"
    embeddings_path = project_folder / "embeddings.json"

    try:
        chunks = read_chunks(chunks_path)
    except FileNotFoundError:
        print(f"Error: chunks.json was not found at {chunks_path}")
        print_summary(0, 0, 0, {}, "FAILURE")
        return 1
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Error reading chunks.json: {error}")
        print_summary(0, 0, 0, {}, "FAILURE")
        return 1

    if not chunks:
        print("Error: chunks.json contains no chunks to embed.")
        print_summary(0, 0, 0, {}, "FAILURE")
        return 1

    try:
        saved_embeddings = load_saved_embeddings(embeddings_path, chunks)
    except (OSError, ValueError) as error:
        print(f"Error reading existing embeddings.json: {error}")
        print_summary(len(chunks), 0, 0, {}, "FAILURE")
        return 1

    # Report empty chunks clearly and do not send them to the API.
    had_error = False
    for chunk in chunks:
        if not chunk["text"].strip():
            print(f"Error: Chunk {chunk['chunk_id']} is empty and cannot be embedded.")
            had_error = True

    # If every nonempty chunk already has a saved embedding, no API key or
    # network call is needed on this run.
    chunks_needing_embeddings = [
        chunk
        for chunk in chunks
        if chunk["text"].strip()
        and chunk["chunk_id"] not in saved_embeddings
    ]

    api_key = None
    client = None
    embedding_config = None

    if chunks_needing_embeddings:
        try:
            # config.py loads the key from .env and rejects an empty or
            # placeholder value. The key itself is never printed.
            from config import GEMINI_API_KEY

            api_key = GEMINI_API_KEY
        except Exception as error:
            print(f"Error loading the Gemini API key: {error}")
            print_summary(len(chunks), 0, 0, saved_embeddings, "FAILURE")
            return 1

        try:
            # Use the official Google GenAI SDK.
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            embedding_config = types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            )
        except Exception as error:
            print(f"Error initializing the Google GenAI client: {safe_error_message(error, api_key)}")
            print_summary(len(chunks), 0, 0, saved_embeddings, "FAILURE")
            return 1

    processed = 0
    newly_generated = 0

    for chunk in chunks:
        processed += 1
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]

        if not chunk_text.strip() or chunk_id in saved_embeddings:
            continue

        try:
            # Generate one vector per chunk so completed results can be saved
            # immediately and reused if a later request fails.
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=chunk_text,
                config=embedding_config,
            )

            response_embeddings = getattr(response, "embeddings", None)
            if not response_embeddings:
                raise ValueError("The API response did not contain an embedding.")

            values = getattr(response_embeddings[0], "values", None)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError("The API response contained an empty embedding vector.")

            vector = [float(value) for value in values]
            if len(vector) != OUTPUT_DIMENSIONALITY or not all(
                math.isfinite(value) for value in vector
            ):
                raise ValueError("The API returned an invalid embedding vector.")

        except Exception as error:
            print(
                f"Error generating embedding for {chunk_id}: "
                f"{safe_error_message(error, api_key)}"
            )
            # Stop here. Existing and already-saved vectors remain available;
            # rerunning the script will resume at this chunk.
            had_error = True
            break

        saved_embeddings[chunk_id] = {
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "embedding": vector,
        }

        try:
            # Save each successful result right away to support resuming later.
            save_embeddings(embeddings_path, chunks, saved_embeddings)
        except OSError as error:
            saved_embeddings.pop(chunk_id, None)
            print(
                f"Error saving embedding for {chunk_id}: "
                f"{safe_error_message(error, api_key)}"
            )
            had_error = True
            break

        newly_generated += 1

    status = "FAILURE" if had_error else "SUCCESS"
    print_summary(len(chunks), processed, newly_generated, saved_embeddings, status)

    # A nonzero exit code makes failures visible to shells and IDEs.
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
