"""Split cleaned shop data into overlapping text chunks."""

import json
from pathlib import Path


# Chunk settings, measured in characters.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def find_chunk_end(text, start):
    """Find a natural boundary near the target chunk size when possible."""
    target_end = min(start + CHUNK_SIZE, len(text))

    # The final chunk can be shorter than the target size.
    if target_end == len(text):
        return target_end

    # Search within 50 characters of the target for a line break first,
    # then a space. This keeps most chunks close to the requested size.
    search_start = max(start + 1, target_end - 50)
    search_end = min(len(text), target_end + 50)
    line_breaks = [
        index + 1
        for index in range(search_start, search_end)
        if text[index] == "\n"
    ]
    if line_breaks:
        return min(line_breaks, key=lambda end: abs(end - target_end))

    spaces = [
        index + 1
        for index in range(search_start, search_end)
        if text[index] == " "
    ]
    if spaces:
        return min(spaces, key=lambda end: abs(end - target_end))

    # If there is no nearby whitespace, use the exact target position.
    return target_end


def make_chunks(text):
    """Create chunks that overlap by the configured number of characters."""
    chunks = []
    start = 0

    while start < len(text):
        end = find_chunk_end(text, start)
        chunk_text = text[start:end]

        # Do not add a chunk if it contains no text.
        if chunk_text:
            chunks.append(chunk_text)

        # Start the next chunk before the previous one ends to keep context.
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP

    return chunks


def main():
    # Read and write files beside this script.
    script_folder = Path(__file__).resolve().parent
    input_path = script_folder / "cleaned_data.txt"
    text_output_path = script_folder / "chunks.txt"
    json_output_path = script_folder / "chunks.json"

    try:
        cleaned_text = input_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return
    except OSError as error:
        print(f"Error: Could not read the input file: {error}")
        return

    chunks = make_chunks(cleaned_text)

    # Give each chunk a unique ID and pair it with its text for the JSON file.
    chunk_records = [
        {"chunk_id": f"chunk_{number:03d}", "text": chunk_text}
        for number, chunk_text in enumerate(chunks, start=1)
    ]

    # Save a readable text version with the chunk ID before each chunk.
    text_output = "\n\n".join(
        f"{record['chunk_id']}\n{record['text']}" for record in chunk_records
    )

    try:
        text_output_path.write_text(text_output, encoding="utf-8")
        json_output_path.write_text(
            json.dumps(chunk_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        print(f"Error: Could not write the chunk files: {error}")
        return

    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")


if __name__ == "__main__":
    main()
