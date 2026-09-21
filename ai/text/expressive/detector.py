"""
Expressive alphabet-extension detection ("soooo", "helloooo", "noooo").

General algorithm, not a hardcoded word list: any run of the same character
repeated 3+ times in a row is treated as an expressive extension (a normal
double letter like "hello" or "look" has a run length of 2 and is left
alone). The base word is the run collapsed down to a single occurrence of
that character.
"""
import re
from typing import Any, Dict

_RUN_PATTERN = re.compile(r"(.)\1{2,}")


def detect_expressive_texting(text: str) -> Dict[str, Any]:
    """
    Returns:
        {
          "detected": bool,
          "extensions": [
            {"original": ..., "base_word": ..., "extended_characters": ..., "extension_count": int},
            ...
          ]
        }
    """
    extensions = []
    if text:
        for word in re.findall(r"\S+", text):
            match = _RUN_PATTERN.search(word)
            if not match:
                continue

            ch = match.group(1)
            run_length = len(match.group(0))
            extended_characters = ch * (run_length - 1)
            base_word = word[: match.start()] + ch + word[match.end():]

            extensions.append({
                "original": word,
                "base_word": base_word,
                "extended_characters": extended_characters,
                "extension_count": run_length - 1,
            })

    return {
        "detected": len(extensions) > 0,
        "extensions": extensions,
    }
