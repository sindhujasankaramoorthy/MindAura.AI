from rapidfuzz import process,fuzz
from vocabulary import load_vocabulary,create_word_set
from preprocess import load_comments
from normalize import normalize_word

from rules import should_skip
#variables
df=load_vocabulary()
word_set=create_word_set(df)
word_counts=dict(zip(df["word"],df["count"]))
#functions required
def rank_candidate(word, candidate, similarity):
    frequency = word_counts.get(candidate, 0)
    score = similarity
    # Frequency bonus
    score += min(frequency / 500, 10)
    # Length penalty
    length_difference = abs(len(word) - len(candidate))
    score -= length_difference * 2
    return score

def correct_word(word):

    info = normalize_word(word)
    word = info["normalized"].lower()

    # Skip words that don't need correction
    if should_skip(word):
        info["corrected"] = word
        return info

    matches = process.extract(
        word,
        word_set,
        scorer=fuzz.WRatio,
        limit=10
    )

    best_word = word
    best_score = -1

    for candidate, similarity, _ in matches:

        score = rank_candidate(
            word,
            candidate,
            similarity
        )

        if score > best_score:
            best_score = score
            best_word = candidate

    info["corrected"] = best_word

    return info
def correct_sentence(sentence):
    words = sentence.split()
    corrected_words = []
    metadata = []
    for word in words:
        info = correct_word(word)
        corrected_words.append(info["corrected"])
        metadata.append(info)
    return {
        "corrected_sentence": " ".join(corrected_words),
        "metadata": metadata
    }
if __name__ == "__main__":
    train, test = load_comments()
    for comment in train["text"].head(10):
        result = correct_sentence(comment)
        print("=" * 80)
        print("Original :", comment)
        print("Corrected:", result["corrected_sentence"])
        print("\nMetadata:")
        for item in result["metadata"]:
            print(item)
        print("=" * 80)
    print(correct_sentence("vera level ippa pesungada mokka nu thalaivaaaaaa"))
    print(correct_sentence("Trailer late ah parthavanga like podunga"))