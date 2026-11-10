"""Load and validate the Gemini API key from the project's .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv


# Load .env from the project folder, regardless of the current working folder.
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Read the key without printing it or including it in error messages.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Treat the setup placeholder as missing until it is replaced with a real key.
if (
    not GEMINI_API_KEY
    or not GEMINI_API_KEY.strip()
    or GEMINI_API_KEY.strip() == "YOUR_API_KEY_HERE"
):
    raise ValueError(
        "GEMINI_API_KEY is missing or still uses the placeholder. "
        "Set it in the project's .env file before using Gemini."
    )
