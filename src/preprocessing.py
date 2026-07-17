import re
import emoji
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (only needs to run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))


def extract_emojis(text):
    """Extract all emojis from a text string as a list."""
    return [char for char in text if char in emoji.EMOJI_DATA]


def remove_emojis(text):
    """Remove emojis from text, leaving only the textual content."""
    return emoji.replace_emoji(text, replace='')


def clean_text(text):
    """Lowercase, remove URLs/mentions/special chars, keep basic punctuation."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)                # remove mentions
    text = re.sub(r"[^a-z0-9\s.,!?']", "", text)     # keep basic chars
    text = re.sub(r"\s+", " ", text).strip()         # collapse whitespace
    return text


def tokenize(text, remove_stopwords=False):
    """Tokenize cleaned text into words."""
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def preprocess_row(text):
    """
    Full pipeline for a single text sample.
    Returns cleaned text, tokens, and extracted emoji list.
    """
    emojis = extract_emojis(text)
    text_no_emoji = remove_emojis(text)
    cleaned = clean_text(text_no_emoji)
    tokens = tokenize(cleaned)
    return {
        "cleaned_text": cleaned,
        "tokens": tokens,
        "emojis": emojis
    }


def preprocess_dataframe(df, text_column="text"):
    """Apply preprocessing to an entire dataframe."""
    results = df[text_column].apply(preprocess_row)
    df["cleaned_text"] = results.apply(lambda x: x["cleaned_text"])
    df["tokens"] = results.apply(lambda x: x["tokens"])
    df["emojis"] = results.apply(lambda x: x["emojis"])
    return df


if __name__ == "__main__":
    # Quick test
    sample = "I am SO happy today!! 😄😄 check this out www.example.com @friend"
    result = preprocess_row(sample)
    print(result)