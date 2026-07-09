import re
from typing import List, Tuple

# Minimal fallback dictionaries (only for unique idioms/phrases that cannot be translated word-by-word)
PHRASE_REPLACEMENTS: List[Tuple[str, str]] = []

# Minimal word replacements mapping key Tanglish tokens to their core semantic English meanings
WORD_REPLACEMENTS = {
    "enaku": "I",
    "enala": "I",
    "ennala": "I",
    "romba": "very",
    "bayama": "afraid",
    "iruku": "feel",
    "mathiri": "somewhat",
    "madhiri": "somewhat",
    "manasu": "mind",
    "nimmadhiya": "calm",
    "thaniya": "alone",
    "purila": "do not understand",
    "panren": "do",
    "kashtama": "difficult",
    "sogama": "sad",
    "kavalaya": "worried",
    "valikuthu": "hurts",
    "thookam": "sleep",
    "sapda": "eat",
    "pesa": "talk",
    "yar": "anyone",
    "kitayum": "with",
    "ethuvum": "anything",
    "ellam": "everything",
    "pidikala": "do not like",
    "pudikula": "do not like",
    "mudiyala": "cannot",
    "mudla": "cannot",
    "oru": "",
    "ah": "",  # adverbial suffix marker
}

def reorder_sov_to_svo(text: str) -> str:
    """
    Applies grammatical reordering from Tamil Subject-Object-Verb (SOV) 
    to English Subject-Verb-Object (SVO) for translated clauses.
    """
    # Split text into clauses based on punctuation
    clauses = re.split(r"([,.:;!?]|\band\b|\bbut\b)", text)
    reordered_clauses = []
    
    for clause in clauses:
        # If it's a delimiter/conjunction, keep it as is
        if re.match(r"^([,.:;!?]|\band\b|\bbut\b|\s+)$", clause):
            reordered_clauses.append(clause)
            continue
            
        stripped = clause.strip()
        if not stripped:
            reordered_clauses.append(clause)
            continue
            
        # 0. Subject at the END of a multi-word predicate (SOV → SVO)
        # e.g., "do not like I" → "I do not like"
        # e.g., "very afraid feel I" → "I feel very afraid"
        # Match: predicate (2+ words) followed by " I" at end
        match_subject_end = re.match(r"^(.+?)\s+I$", stripped, re.IGNORECASE)
        if match_subject_end:
            predicate = match_subject_end.group(1)
            stripped = f"I {predicate}"
            
        # Grammatical reordering patterns
        # 1. Subject [X] feel -> Subject feel [X]
        match_feel = re.match(r"^I\s+(.+?)\s+feel$", stripped, re.IGNORECASE)
        if match_feel:
            middle = match_feel.group(1)
            stripped = f"I feel {middle}"
            
        # 1b. No subject: [X] feel -> feel [X]
        else:
            match_feel_no_sub = re.match(r"^(.+?)\s+feel$", stripped, re.IGNORECASE)
            if match_feel_no_sub:
                middle = match_feel_no_sub.group(1)
                stripped = f"feel {middle}"
                
            # 2. Subject [X] do/doing -> Subject do [X] / Subject am doing [X]
            else:
                match_do = re.match(r"^I\s+(.+?)\s+do$", stripped, re.IGNORECASE)
                if match_do:
                    middle = match_do.group(1)
                    if middle.lower().endswith("ing"):
                        stripped = f"I am {middle}"
                    else:
                        stripped = f"I do {middle}"
                        
                # 3. Subject [X] cannot -> Subject cannot [X]
                else:
                    match_cannot = re.match(r"^I\s+(.+?)\s+cannot$", stripped, re.IGNORECASE)
                    if match_cannot:
                        middle = match_cannot.group(1)
                        stripped = f"I cannot {middle}"
                        
                    # 4. Subject [X] do not understand -> Subject do not understand [X]
                    else:
                        match_understand = re.match(r"^I\s+(.+?)\s+do not understand$", stripped, re.IGNORECASE)
                        if match_understand:
                            middle = match_understand.group(1)
                            stripped = f"I do not understand {middle}"
            
        reordered_clauses.append(clause.replace(clause.strip(), stripped))
        
    return "".join(reordered_clauses)

def normalize_tanglish_semantics(text: str) -> str:
    """
    Convert common Tamil-English journaling expressions to natural English meanings.
    Uses structural/grammatical transformations rather than large hardcoded phrase maps.
    """
    processed = text.lower()
    
    # 1. Structural Patterns (applied before word-by-word)
    
    # "enaku [oru mathiri] mind [adj] ah iruku" -> "I feel mentally [adj]"
    processed = re.sub(
        r"\benaku\s+(?:oru\s+ma[dt]hiri\s+)?mind\s+(\w+)\s+ah\s+iruku\b",
        r"I feel mentally \1",
        processed
    )
    
    # "mind [adj] ah iruku" -> "my mind is [adj]"
    processed = re.sub(
        r"\bmind\s+(\w+)\s+ah\s+iruku\b",
        r"my mind is \1",
        processed
    )
    
    # "enala/ennala ethuvum panna mudiyala/mudla" -> "I cannot do anything"
    processed = re.sub(
        r"\b(?:e|en)nala\s+ethuvum\s+(?:panna\s+)?(?:mudiyala|mudla)\b",
        "I cannot do anything",
        processed
    )
    
    # "enala/ennala/enaku mudiyala/mudla" -> "I cannot handle it"
    processed = re.sub(
        r"\b(?:e|en)nala\s+(?:mudiyala|mudla)\b",
        "I cannot handle it",
        processed
    )
    processed = re.sub(
        r"\benaku\s+(?:mudiyala|mudla)\b",
        "I cannot handle it",
        processed
    )
    
    # "[verb] panna mudiyala/mudla" -> "cannot [verb]"
    # e.g., "work panna mudiyala" -> "cannot work"
    # e.g., "focus panna mudiyala" -> "cannot focus"
    # e.g., "concentrate panna mudiyala" -> "cannot concentrate"
    processed = re.sub(
        r"\bethuvum\s+panna\s+(?:mudiyala|mudla)\b",
        "cannot do anything",
        processed
    )
    processed = re.sub(
        r"\b(\w+)\s+panna\s+(?:mudiyala|mudla)\b",
        r"cannot \1",
        processed
    )
    
    # "[verb]ve mudiyala/mudla" -> "cannot [verb]"
    # e.g., "thoongave mudiyala" -> "cannot sleep"
    def replace_verb_ve(match):
        verb = match.group(1)
        verb_map = {
            "thoonga": "sleep",
            "sapda": "eat",
            "pesa": "talk",
        }
        english_verb = verb_map.get(verb, verb)
        return f"cannot {english_verb}"
        
    processed = re.sub(
        r"\b(\w+)ve\s+(?:mudiyala|mudla)\b",
        replace_verb_ve,
        processed
    )
    
    # "[verb] thonala" -> "do not feel like [verb-ing]"
    # e.g., "sapda thonala" -> "do not feel like eating"
    def replace_verb_thonala(match):
        verb = match.group(1)
        verb_ing_map = {
            "sapda": "eating",
            "pesa": "talking",
            "iruka": "being",
        }
        return f"do not feel like {verb_ing_map.get(verb, verb)}"
        
    processed = re.sub(
        r"\b(\w+)\s+thonala\b",
        replace_verb_thonala,
        processed
    )
    
    # "thaniya iruka pidikala" -> "do not like being alone"
    processed = re.sub(
        r"\bthaniya\s+iruka\s+pidikala\b",
        "do not like being alone",
        processed
    )
    
    # "thookam varala/varla" -> "cannot sleep"
    processed = re.sub(
        r"\bthookam\s+(?:varala|varla)\b",
        "cannot sleep",
        processed
    )
    
    # "nenachu nenachu tired agiten" -> "tired from thinking repeatedly"
    processed = re.sub(
        r"\bnenachu\s+nenachu\s+tired\s+agiten\b",
        "tired from thinking repeatedly",
        processed
    )
    
    # "manasu kashtama iruku" -> "my heart feels heavy"
    processed = re.sub(
        r"\bmanasu\s+kashtama\s+iruku\b",
        "my heart feels heavy",
        processed
    )

    # 2. Word-by-word replacements
    # Sort keys by length descending to prevent substring matching
    sorted_words = sorted(WORD_REPLACEMENTS.items(), key=lambda x: len(x[0]), reverse=True)
    for pattern, replacement in sorted_words:
        if replacement == "":
            processed = re.sub(r"\b" + re.escape(pattern) + r"\b", "", processed)
        else:
            processed = re.sub(r"\b" + re.escape(pattern) + r"\b", replacement, processed)
            
    # Clean up duplicate spaces
    processed = re.sub(r"\s+", " ", processed).strip()
    
    # 3. Apply grammatical SVO reordering
    processed = reorder_sov_to_svo(processed)
    
    return re.sub(r"\s+", " ", processed).strip()
