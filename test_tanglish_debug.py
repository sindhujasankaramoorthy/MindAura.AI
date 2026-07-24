import os
import sys
import traceback

print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)
print("--------------------------------------------------\n")

words = [
    "enaku",
    "romba",
    "kastama",
    "iruku",
    "pudikula",
    "mudiyala",
    "theriyum",
    "konjum"
]

print("=== Tanglish Vocabulary Test ===")
try:
    # is_tanglish_word does not exist in vocabulary.py directly.
    # The actual vocabulary check uses the word set.
    from ai.tanglish_model.src.vocabulary import load_vocabulary, create_word_set
    
    df = load_vocabulary()
    word_set = create_word_set(df)
    
    def is_tanglish_word(word):
        return word.lower() in word_set

    for word in words:
        print(f"{word} -> {is_tanglish_word(word)}")

except Exception as e:
    print("Failed to import or use vocabulary:")
    traceback.print_exc()

print("\n=== Tanglish Correction Test ===")
try:
    from ai.tanglish_model.src.autocorrect import correct_word
    
    for word in words:
        result = correct_word(word)
        print(f"{word} -> {result.get('corrected', word)}")

except Exception as e:
    print("Failed to import or use autocorrect:")
    traceback.print_exc()
