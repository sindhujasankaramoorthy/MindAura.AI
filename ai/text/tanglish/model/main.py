from src.preprocess import (
    load_vocabulary,
    explore_vocabulary,
    clean_vocabulary,
    build_autocorrect_vocabulary,
    analyze_vocabulary,
    save_vocabulary,
    load_comments,
    explore_comments
)
def main():

    # -------------------------
    # Vocabulary
    # -------------------------

    df = load_vocabulary()

    print("BEFORE CLEANING")
    explore_vocabulary(df)

    # Clean the vocabulary
    df = clean_vocabulary(df)

    print("\nAFTER CLEANING")
    explore_vocabulary(df)

    # Build vocabulary specifically for autocorrect
    auto_df = build_autocorrect_vocabulary(df)

    print("\nAUTOCORRECT VOCABULARY")
    analyze_vocabulary(auto_df)

    # Save the filtered vocabulary
    save_vocabulary(auto_df)

    print("\nAutocorrect vocabulary saved successfully!")

    # -------------------------
    # Comments Dataset
    # -------------------------

    test, train = load_comments()

    print("\n========== TRAIN DATASET ==========")
    explore_comments(train)

    print("\n========== TEST DATASET ==========")
    explore_comments(test)


if __name__ == "__main__":
    main()