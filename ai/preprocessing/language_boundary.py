import logging
from wordfreq import zipf_frequency as z

logger = logging.getLogger(__name__)

# Very common Tanglish words that might overlap with obscure English words
TANGLISH_PARTICLES = {
    "na", "da", "di", "ah", "eh", "la", "ma", "pa", "ka", "ta", "ni",
    "un", "en", "oru", "nee", "athu", "ithu", "ethu", "adhu", "idhu",
    "kuda", "kooda", "po", "vaa", "tha", "edu",
    "romba", "konjam", "nalla", "illa", "ama", "aama",
    "vanda", "poda", "saptiya", "pandra", "panra", "panran", "pandran",
    "iruku", "irukku", "enaku", "enakku", "unaku", "unakku",
    "rendu", "perum", "peru", "varum", "tharum",
    "ku", "nu", "inga", "anga", "enga",
    "avan", "aval", "avar", "avanga", "ivanga"
}

# Strong English words that often get absorbed by the Tanglish vocabulary due to code-mixing
STRONG_ENGLISH_WORDS = {
    "and", "the", "is", "in", "to", "of", "it", "that", "you", "for",
    "on", "with", "as", "this", "but", "his", "they", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good",
    "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "are", "don't", "dont", "do", "does", "doesn't",
    "fights", "fight", "fighting", "fought", "fighted", "hates", "hate",
    "hating", "loves", "love", "loving", "cries", "cry", "crying",
    "likes", "liking", "liked", "dislike", "dislikes", "disliked",
    "sad", "happy", "angry", "mad", "upset", "glad", "tired", "stressed"
}

def classify_token_language(token: str) -> str:
    """
    Intelligently determines if a token is strictly English.
    Returns 'ENGLISH' if true, otherwise 'UNKNOWN' to fall through
    to Tanglish classification.
    """
    w_lower = token.lower()

    if w_lower in TANGLISH_PARTICLES:
        return "UNKNOWN"

    if w_lower in STRONG_ENGLISH_WORDS:
        return "ENGLISH"

    # High frequency English words (Zipf frequency > 3.0 is highly common in English)
    freq = z(w_lower, 'en')
    if freq >= 3.0:
        return "ENGLISH"
        
    # Check for title case (likely a name or proper noun)
    if token.istitle() and freq >= 1.0:
        return "ENGLISH"
        
    return "UNKNOWN"
