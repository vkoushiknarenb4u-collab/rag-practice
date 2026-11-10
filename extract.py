"""Extract text from Shop_Knowledge_Base.pdf into a text file."""

from pathlib import Path

try:
    import pymupdf
except ImportError:
    print("PyMuPDF is not installed. Install it with: pip install pymupdf")
    raise SystemExit(1)


def main():
    # Look beside this script first, then in the Downloads folder where the
    # supplied PDF is currently located.
    script_folder = Path(__file__).resolve().parent
    pdf_path = script_folder / "Shop_Knowledge_Base.pdf"
    if not pdf_path.is_file():
        pdf_path = Path.home() / "Downloads" / "Shop_Knowledge_Base.pdf"

    # Save the extracted text beside this script.
    output_path = script_folder / "extracted_data.txt"

    if not pdf_path.is_file():
        print(f"Error: PDF file not found: {pdf_path}")
        return

    try:
        # Open the PDF and collect text from every page.
        with pymupdf.open(pdf_path) as pdf_document:
            all_pages_text = [page.get_text() for page in pdf_document]

        # Join page text with a newline so page contents stay separated.
        extracted_text = "\n".join(all_pages_text)

        # Write the combined text to a UTF-8 text file.
        output_path.write_text(extracted_text, encoding="utf-8")

        print(f"Success! Extracted text saved to: {output_path}")
        print(f"Total characters extracted: {len(extracted_text)}")
    except (pymupdf.FileDataError, OSError, RuntimeError) as error:
        print(f"Error: Could not read or process the PDF: {error}")


if __name__ == "__main__":
    main()
