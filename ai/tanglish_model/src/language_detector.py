from wordfreq import zipf_frequency

def is_english(word):
    """
    Returns True if the word is a common English word.
    """
    return zipf_frequency(word.lower(), "en") > 3
if __name__ == "__main__":
    words = [
        "trailer",
        "level",
        "hello",
        "romba",
        "epdi",
        "iruku"
    ]

    for w in words:
        print(w, "->", is_english(w))