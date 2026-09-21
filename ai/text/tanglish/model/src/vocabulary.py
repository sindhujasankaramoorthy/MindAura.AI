import pandas as pd
from pathlib import Path
def load_vocabulary():
    """Load the cleaned Tanglish vocabulary."""
    base_dir=Path(__file__).resolve().parent.parent
    csv_path=base_dir/"data"/"processed"/"tanglish_words.csv"
    df = pd.read_csv(csv_path)
    return df
def create_word_set(df):
    """Convert the vocabulary into a Python set for very fast word lookup."""
    return set(df["word"])
def create_frequency_dict(df):
    """Create a dictionary: word -> frequency"""
    return dict(zip(df["word"], df["count"]))