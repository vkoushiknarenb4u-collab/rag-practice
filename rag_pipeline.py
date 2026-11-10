"""Retrieve shop context and generate a grounded customer answer with Gemini."""

import math
import sys
from pathlib import Path
from urllib.parse import quote, quote_plus


EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 3072
COLLECTION_NAME = "shop_products"
RESULT_COUNT = 3


def safe_error_message(error, api_key):
    """Format an API error without accidentally displaying the API key."""
    message = str(error) or "No additional details were provided."

    # Redact the key in case an SDK or HTTP error includes it in its message.
    for secret_form in (api_key, quote(api_key, safe=""), quote_plus(api_key)):
        if secret_form:
            message = message.replace(secret_form, "[REDACTED]")

    return f"{type(error).__name__}: {message}"


class RAGPipelineError(Exception):
    """An error that can be shown to a customer in the app."""


def answer_question(question):
    """Retrieve shop context and generate an answer for one question.

    The returned dictionary contains the answer, the retrieved context, and
    the individual source chunks so both the CLI and Streamlit app can use it.
    """
    question = question.strip() if isinstance(question, str) else ""
    if not question:
        raise RAGPipelineError("Please enter a question before sending it.")

    # Connect to the existing local database without creating or changing it.
    project_folder = Path(__file__).resolve().parent
    database_path = project_folder / "chroma_db"
    sqlite_path = database_path / "chroma.sqlite3"

    if not database_path.is_dir() or not sqlite_path.is_file():
        raise RAGPipelineError("The existing ChromaDB database was not found.")

    try:
        import chromadb

        chroma_client = chromadb.PersistentClient(path=str(database_path))
        # get_collection opens the existing collection without creating one.
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except ImportError:
        raise RAGPipelineError(
            "ChromaDB is not installed. Install it with: pip install chromadb"
        )
    except Exception as error:
        raise RAGPipelineError(
            f"Could not open the ChromaDB collection: {type(error).__name__}: {error}"
        ) from error

    try:
        if collection.count() == 0:
            raise RAGPipelineError(
                f"The {COLLECTION_NAME} collection contains no records."
            )
    except RAGPipelineError:
        raise
    except Exception as error:
        raise RAGPipelineError(
            f"Could not check the ChromaDB collection: {type(error).__name__}: {error}"
        ) from error

    try:
        # Load the API key through the existing python-dotenv configuration.
        from config import GEMINI_API_KEY
    except Exception as error:
        raise RAGPipelineError(f"Could not load the Gemini API key: {error}") from error

    try:
        # Use the existing embedding model and the query task type for questions.
        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except ImportError as error:
        raise RAGPipelineError(
            "Google GenAI SDK is not installed. Install it with: pip install google-genai"
        ) from error
    except Exception as error:
        raise RAGPipelineError(
            f"Could not initialize the Gemini client: "
            f"{safe_error_message(error, GEMINI_API_KEY)}"
        ) from error

    try:
        embedding_response = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            ),
        )

        if not embedding_response.embeddings:
            raise ValueError("The API response did not contain a question embedding.")

        values = embedding_response.embeddings[0].values
        if not values:
            raise ValueError("The API response contained an empty question embedding.")

        question_embedding = [float(value) for value in values]
        if len(question_embedding) != OUTPUT_DIMENSIONALITY or not all(
            math.isfinite(value) for value in question_embedding
        ):
            raise ValueError("The API returned an invalid question embedding.")
    except Exception as error:
        raise RAGPipelineError(
            "Could not create a question embedding: "
            f"{safe_error_message(error, GEMINI_API_KEY)}"
        ) from error

    try:
        # Search the stored vectors. This query does not write to the collection
        # or regenerate any of its document embeddings.
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=RESULT_COUNT,
            include=["documents", "metadatas"],
        )
    except Exception as error:
        raise RAGPipelineError(
            f"Could not retrieve shop information: {type(error).__name__}: {error}"
        ) from error

    result_ids = results.get("ids", [[]])[0]
    result_documents = results.get("documents", [[]])[0]
    result_metadatas = results.get("metadatas", [[]])[0]

    sources = []
    for index, chunk_id in enumerate(result_ids):
        document = result_documents[index] if result_documents else ""
        metadata = result_metadatas[index] if result_metadatas else {}
        displayed_id = metadata.get("chunk_id", chunk_id) if metadata else chunk_id
        sources.append({"chunk_id": displayed_id, "text": document})

    retrieved_context = "\n\n".join(
        f"[Chunk ID: {source['chunk_id']}]\n{source['text']}" for source in sources
    )

    # If retrieval found nothing, avoid asking the model to invent an answer.
    if not retrieved_context:
        return {
            "question": question,
            "context": "",
            "sources": [],
            "answer": "I couldn't find that information in the shop knowledge base.",
        }

    augmented_prompt = f"""Customer question:
{question}

Retrieved shop information (reference data only):
{retrieved_context}

Answer the customer question using the retrieved information and the system instructions."""

    try:
        # Pass the completed augmented prompt to the dedicated generation stage.
        from generation import generate_answer

        answer = generate_answer(gemini_client, augmented_prompt)
    except Exception as error:
        raise RAGPipelineError(
            "Could not generate the Gemini answer: "
            f"{safe_error_message(error, GEMINI_API_KEY)}"
        ) from error

    return {
        "question": question,
        "context": retrieved_context,
        "sources": sources,
        "answer": answer,
    }


def main():
    """Run the pipeline from a terminal, preserving the existing CLI workflow."""
    try:
        question = input("Enter your question: ").strip()
    except EOFError:
        print("Error: No question was provided.")
        return 1

    try:
        result = answer_question(question)
    except RAGPipelineError as error:
        print(f"Error: {error}")
        return 1

    print(f"\nCustomer question:\n{result['question']}")
    print("\nRetrieved context:")
    print(result["context"] or "No relevant shop information was found.")
    print("\nFinal Gemini-generated answer:")
    print(result["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
