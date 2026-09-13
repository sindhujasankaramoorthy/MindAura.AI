import pandas as pd
from pathlib import Path

#the required functions
PROJECT_ROOT=Path(__file__).resolve().parent.parent
def load_vocabulary():
    path=PROJECT_ROOT/"data"/"raw"/"tanglish_words.csv"
    df=pd.read_csv(path)
    return df
def explore_vocabulary(df):
    print("duplicate:")
    print(df.duplicated().sum())
    print("duplicate:word")
    print(df["word"].duplicated().sum())
    print("missing:")
    print(df.isnull().sum())
def clean_vocabulary(df):
    df=df.dropna()#row with missing values
    df=df.drop_duplicates(subset=["word"])
    df["word"]=df["word"].str.lower()
    df["word"]=df["word"].str.strip()
    return df
def analyze_vocabulary(df):
    print("\n========== VOCABULARY ANALYSIS ==========")

    print("\nTop 100 Most Frequent Words")
    print(df.sort_values(by="count", ascending=False).head(100))

    print("\nWords with length <= 2")
    short_words = df[df["word"].str.len() <= 2]
    print(short_words.head(100))
    print("Total:", len(short_words))

    print("\nWords containing digits")
    digits = df[df["word"].str.contains(r"\d", regex=True)]
    print(digits.head(50))
    print("Total:", len(digits))

    print("\nWords containing non-alphabet characters")
    special = df[~df["word"].str.match(r"^[a-zA-Z]+$")]
    print(special.head(50))
    print("Total:", len(special))
def build_autocorrect_vocabulary(df):
    """
    Build a clean vocabulary for the autocorrect model.
    """

    print("\n========== BUILDING AUTOCORRECT VOCABULARY ==========")

    original_size = len(df)

    # Keep only alphabetic words
    df = df[df["word"].str.match(r"^[a-zA-Z]+$")]

    # Remove words containing digits
    df = df[~df["word"].str.contains(r"\d", regex=True)]

    # Remove obvious junk/web words
    remove_words = {
        "href",
        "amp",
        "com",
        "www",
        "http",
        "https",
        "youtube",
        "search",
        "searchquery",
        "search_query"
    }

    df = df[~df["word"].isin(remove_words)]

    print(f"Original vocabulary size : {original_size}")
    print(f"Filtered vocabulary size : {len(df)}")
    print(f"Removed                 : {original_size - len(df)}")

    print("\nFirst 20 words")
    print(df.head(20))

    return df
def save_vocabulary(df):
    path = PROJECT_ROOT / "data" / "processed" / "autocorrect_vocab.csv"
    df.to_csv(path, index=False)
#comments
def load_comments():
    train_path = PROJECT_ROOT / "data" / "raw" / "train.tsv"
    test_path = PROJECT_ROOT / "data" / "raw" / "test.tsv"
    train = pd.read_csv(train_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")
    return train, test
def explore_comments(df):
    """Display information about the comments dataset."""
    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 5 Rows:")
    print(df.head())
#main
def main():
    df = load_vocabulary()
    print("BEFORE CLEANING")
    explore_vocabulary(df)
    df = clean_vocabulary(df)
    print("\nAFTER CLEANING")
    explore_vocabulary(df)
    analyze_vocabulary(df)
    # NEW STEP
    auto_df = build_autocorrect_vocabulary(df)
    # Save ONLY the autocorrect vocabulary
    save_vocabulary(auto_df)
    print("\nAutocorrect vocabulary saved successfully!")
if __name__=="__main__":
    main()
