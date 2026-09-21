import re
def has_elongation(word):
    """
    Returns True if a character repeats 3 or more times consecutively.
    """
    return bool(re.search(r"(.)\1{2,}", word))
def detect_elongation(word):
    """
    Detect if a character repeats 3 or more times consecutively.

    Returns:
        (elongated, letter, repeat_count)
    """

    match = re.search(r"(.)\1{2,}", word)

    if match:
        repeated = match.group(0)      # e.g. "iiii"
        letter = match.group(1)        # e.g. "i"
        count = len(repeated)          # e.g. 4

        return True, letter, count

    return False, None, 0
def normalize_elongation(word):
    """
    Reduce repeated letters.

    Examples:
    epdiiii   -> epdi
    rombaaaa  -> romba
    seriiiiii -> seri
    """
    return re.sub(r"(.)\1{2,}", r"\1", word)
def normalize_word(word):

    elongated, letter, count = detect_elongation(word)

    normalized = normalize_elongation(word)

    return {
        "original": word,
        "normalized": normalized,
        "metadata": {
            "elongated": elongated,
            "letter": letter,
            "repeat_count": count
        }
    }
if __name__ == "__main__":
    words = [
        "romba",
        "rombaa",
        "rombaaaa",
        "epdi",
        "epdiii",
        "seri",
        "seriiiiii",
        "amma",
        "ammaaaa"
    ]

    for w in words:
        print(f"{w:12} -> {normalize_word(w)}")