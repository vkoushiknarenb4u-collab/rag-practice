"""Clean extracted shop text by normalizing whitespace and formatting."""

from pathlib import Path
import re


def clean_text(text):
    """Remove unwanted control characters and normalize whitespace."""
    # Remove nonprinting ASCII control characters, while keeping tabs and
    # line breaks so the text structure can be cleaned up below.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # A byte order mark is a file-format marker, not part of the shop content.
    text = text.replace("\ufeff", "")

    cleaned_lines = []
    blank_line_pending = False

    # Process each line separately so product fields and descriptions remain
    # on their original lines.
    for line in text.splitlines():
        # Replace repeated spaces, tabs, and other horizontal whitespace with
        # one regular space, then trim the line edges.
        line = re.sub(r"[^\S\r\n]+", " ", line).strip()

        if line:
            # Keep at most one blank line between content sections.
            if blank_line_pending and cleaned_lines:
                cleaned_lines.append("")
            cleaned_lines.append(line)
            blank_line_pending = False
        elif cleaned_lines:
            blank_line_pending = True

    return "\n".join(cleaned_lines)


def main():
    # Read and write files beside this script.
    script_folder = Path(__file__).resolve().parent
    input_path = script_folder / "extracted_data.txt"
    output_path = script_folder / "cleaned_data.txt"

    try:
        original_text = input_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(original_text)
        output_path.write_text(cleaned_text, encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return
    except OSError as error:
        print(f"Error: Could not read or write a file: {error}")
        return

    print(f"Cleaned text saved to: {output_path}")
    print(f"Original character count: {len(original_text)}")
    print(f"Cleaned character count: {len(cleaned_text)}")


if __name__ == "__main__":
    main()
