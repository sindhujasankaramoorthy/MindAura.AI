import re
import string
from language_detector import is_english
def is_placeholder(word):
    """
    Detect placeholders like:
    [PERSON]
    [PLACE]
    [ORG]
    """
    return bool(re.fullmatch(r"\[[A-Z_]+\]", word))
def is_number(word):
    """
    Detect numbers.
    """
    return bool(re.fullmatch(r"[0-9]+([.,][0-9]+)?('?s)?", word))
def is_url(word):
    """
    Detect URLs.
    """
    return bool(
        re.match(
            r"^(http://|https://|www\.)",
            word.lower()
        )
    )
def is_email(word):
    """
    Detect email addresses.
    """
    return bool(
        re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            word
        )
    )
def is_hashtag(word):
    return word.startswith("#")
def is_mention(word):
    return word.startswith("@")
def is_punctuation(word):
    return all(ch in string.punctuation for ch in word)
def is_short_word(word):
    return len(word) <= 2
def should_skip(word):
    """
    Skip words that should NOT be autocorrected.
    """

    return (
        is_placeholder(word)
        or is_number(word)
        or is_url(word)
        or is_email(word)
        or is_hashtag(word)
        or is_mention(word)
        or is_punctuation(word)
        or is_short_word(word)
        or is_english(word)
    )