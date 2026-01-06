import re
import emoji
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    1. Removing HTML tags.
    2. Demojizing emojis (converting to text representation) or removing them. 
       Note: For fake news detection, sometimes emojis are a signal, so we'll keep them as text.
    3. Normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Handle Emojis (demojize converts 🐍 to :snake:)
    text = emoji.demojize(text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
