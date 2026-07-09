"""
NERProtection — Dynamic zero-shot Named Entity Recognition layer for MindAura.

Uses GLiNER (NeuML/gliner-bert-tiny) to detect PERSON, GPE, ORG, LOC, and
relationship entities without any hardcoded name/place/org dictionaries.

Falls back transparently to dslim/bert-base-NER-uncased if GLiNER is unavailable.

API is fully backward-compatible:
    detect_entities(text) -> List[(start, end, type, word)]
    protect(text, entities) -> (protected_text, placeholder_map)
    restore(text, placeholder_map) -> str
Journel entry -> text_normalizer -> language_detector
"""

import re
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Small safety-net for relationship words that GLiNER may miss in
# very short / context-poor sentences (e.g. "mom said").
# This is NOT a name / place list — purely grammatical role words.
# ──────────────────────────────────────────────────────────────────
_RELATION_FALLBACK = re.compile(
    r'\b(mom|mother|dad|father|sister|brother|grandma|grandmother|'
    r'grandpa|grandfather|uncle|aunt|cousin|wife|husband|spouse|'
    r'partner|son|daughter|friend|girlfriend|boyfriend|parent|'
    r'parents|sibling|siblings|fiance|fiancee)\b',
    re.IGNORECASE
)

# GLiNER zero-shot entity labels
_GLINER_LABELS = [
    "person",
    "location",
    "organization",
    "geopolitical entity",
    "relationship",
    "family member",
]


class NERProtection:
    """
    NERProtection Layer for MindAura.

    Dynamically detects PERSON, GPE/LOC, ORG, and relationship entities
    using a zero-shot GLiNER model so no hardcoded name/place dictionaries
    are needed.  A tiny regex fallback catches family-role words that may be
    missed in single-word / context-poor inputs.

    Entities are masked with **TYPE_N** placeholders during correction and
    restored afterward, preserving original casing.
    """

    # Mapping from GLiNER label → canonical type string used downstream
    _LABEL_MAP = {
        "person":             "PERSON",
        "location":           "GPE",
        "organization":       "ORG",
        "geopolitical entity": "GPE",
        "relationship":       "RELATION",
        "family member":      "RELATION",
    }

    def __init__(self):
        self._gliner: Optional[object] = None
        self._bert_ner = None

        # --- Attempt 1: GLiNER tiny ---
        try:
            from gliner import GLiNER  # type: ignore
            logger.info("Initializing NER Protection Layer (NeuML/gliner-bert-tiny)…")
            self._gliner = GLiNER.from_pretrained("NeuML/gliner-bert-tiny")
            logger.info("GLiNER loaded successfully.")
        except Exception as exc:
            logger.warning(f"GLiNER unavailable ({exc}).")

        # --- Attempt 2: BERT-NER ---
        try:
            from transformers import pipeline  # type: ignore
            logger.info("Loading NER model dslim/bert-base-NER-uncased…")
            self._bert_ner = pipeline("ner", model="dslim/bert-base-NER-uncased")
            logger.info("BERT NER model loaded successfully.")
        except Exception as exc:
            logger.warning(f"BERT NER model unavailable ({exc}).")

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _detect_gliner(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Run GLiNER zero-shot NER and return (start, end, TYPE, word) tuples."""
        predictions = self._gliner.predict_entities(
            text,
            _GLINER_LABELS,
            threshold=0.45,
        )
        entities = []
        for pred in predictions:
            label = pred.get("label", "").lower()
            ent_type = self._LABEL_MAP.get(label)
            if ent_type is None:
                continue
            start = pred["start"]
            end   = pred["end"]
            word  = text[start:end]
            entities.append((start, end, ent_type, word))
        return entities

    def _detect_bert(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Run BERT-NER fallback and return (start, end, TYPE, word) tuples."""
        _label_map = {"PER": "PERSON", "LOC": "GPE", "ORG": "ORG"}
        MIN_CONF = 0.75
        raw = self._bert_ner(text)
        entities: List[Dict] = []
        for r in raw:
            ent_key = r["entity"].split("-")[-1]
            if ent_key not in _label_map:
                continue
            if float(r.get("score", 1.0)) < MIN_CONF:
                continue
            entities.append({
                "start": r["start"], "end": r["end"],
                "type": _label_map[ent_key],
                "word": text[r["start"]:r["end"]],
            })

        # Sort → merge consecutive same-type
        entities.sort(key=lambda x: x["start"])
        merged: List[Dict] = []
        for ent in entities:
            if merged and ent["type"] == merged[-1]["type"] and ent["start"] <= merged[-1]["end"] + 1:
                merged[-1]["end"] = max(merged[-1]["end"], ent["end"])
                merged[-1]["word"] = text[merged[-1]["start"]:merged[-1]["end"]]
            else:
                merged.append(ent)

        return [(m["start"], m["end"], m["type"], m["word"]) for m in merged]

    @staticmethod
    def _align_to_word_boundaries(
        text: str,
        entities: List[Tuple[int, int, str, str]],
    ) -> List[Tuple[int, int, str, str]]:
        """
        Filter out sub-word fragment spans.  An entity must start and end
        exactly at whitespace-delimited word boundaries.
        """
        word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        result = []
        for start, end, etype, word in entities:
            is_word_start = any(ws == start for ws, _ in word_spans)
            is_word_end   = any(we == end   for _, we in word_spans)
            if is_word_start and is_word_end:
                result.append((start, end, etype, word))
            else:
                logger.debug(f"Skipping sub-word NER fragment: '{word}' at {start}:{end}")
        return result

    @staticmethod
    def _reject_protected(
        entities: List[Tuple[int, int, str, str]],
    ) -> List[Tuple[int, int, str, str]]:
        """
        Reject entities that are known negation, emotional vocabulary,
        or Tanglish words, so they aren't mistakenly masked.
        """
        try:
            from .advanced_correction import PROTECTED_WORDS, NEGATION_RECOVERY_MAP, CUSTOM_OVERRIDES
            from .tanglish_patterns import WORD_REPLACEMENTS
            _reject = (
                {w.lower() for w in PROTECTED_WORDS}
                | {k.lower() for k in NEGATION_RECOVERY_MAP}
                | {k.lower() for k in CUSTOM_OVERRIDES}
                | {k.lower() for k in WORD_REPLACEMENTS}
                | {"oru", "ah", "dha", "sol", "yen", "nee", "en", "un", "da", "di", "la", "ve", "iruku", "mudla", "mudiyala"}
            )
        except Exception:
            _reject = set()

        filtered = []
        for s, e, t, w in entities:
            words = [word.lower() for word in re.findall(r"\w+", w)]
            if any(word in _reject for word in words):
                logger.debug(f"Rejecting entity '{w}' due to containing protected/Tanglish word")
                continue
            filtered.append((s, e, t, w))
        return filtered

    @staticmethod
    def _add_relation_fallback(
        text: str,
        existing: List[Tuple[int, int, str, str]],
    ) -> List[Tuple[int, int, str, str]]:
        """
        Append family-role words not already covered by the model.
        """
        result = list(existing)
        for m in _RELATION_FALLBACK.finditer(text):
            s, e = m.span()
            overlaps = any(not (e <= es or s >= ee) for es, ee, _, _ in result)
            if not overlaps:
                result.append((s, e, "RELATION", m.group(0)))
        return result

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def detect_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Detect named entities (PERSON, GPE, ORG, RELATION) in *text*.

        Returns a sorted list of ``(start, end, type, original_word)`` tuples.
        """
        raw = []
        if self._bert_ner is not None:
            raw.extend(self._detect_bert(text))
        if self._gliner is not None:
            gliner_raw = self._detect_gliner(text)
            for g_ent in gliner_raw:
                gs, ge, gt, gw = g_ent
                # Avoid overlapping with already detected BERT entities
                overlaps = any(not (ge <= bs or gs >= be) for bs, be, _, _ in raw)
                if not overlaps:
                    raw.append(g_ent)

        # Post-processing pipeline (order matters)
        raw = self._align_to_word_boundaries(text, raw)
        raw = self._reject_protected(raw)
        raw = self._add_relation_fallback(text, raw)

        raw.sort(key=lambda x: x[0])

        if raw:
            ner_log = "\n".join(
                f"  - {t}: '{w}' (at {s}:{e})" for s, e, t, w in raw
            )
            logger.debug(f"NER Detected:\n{ner_log}")
        else:
            logger.debug("NER Detected: []")

        return raw

    def protect(
        self,
        text: str,
        entities: List[Tuple[int, int, str, str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Replace detected entities with ``**TYPE_N**`` placeholders.

        Returns ``(protected_text, placeholder_map)``.
        """
        if entities is None:
            entities = self.detect_entities(text)

        if not entities:
            return text, {}

        unique_to_placeholder: Dict[Tuple, str] = {}
        type_counters: Dict[str, int] = {}
        replacements = []

        for start, end, ent_type, original_word in entities:
            key = (ent_type, original_word.lower())
            if key not in unique_to_placeholder:
                type_counters[ent_type] = type_counters.get(ent_type, 0) + 1
                placeholder = f"**{ent_type}_{type_counters[ent_type]}**"
                unique_to_placeholder[key] = placeholder
            else:
                placeholder = unique_to_placeholder[key]
            replacements.append((start, end, placeholder, original_word))

        # Right-to-left to avoid offset shifts
        replacements.sort(key=lambda x: x[0], reverse=True)

        protected = text
        placeholder_map: Dict[str, str] = {}
        for start, end, placeholder, original_word in replacements:
            protected = protected[:start] + placeholder + protected[end:]
            placeholder_map[placeholder] = original_word

        logger.debug(f"Protected Text: '{protected}'")
        return protected, placeholder_map

    def restore(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore ``**TYPE_N**`` placeholders back to original words, preserving/correcting casing for proper nouns."""
        restored = text
        for placeholder, original in placeholder_map.items():
            match = re.match(r"\*\*([A-Z_]+)_\d+\*\*", placeholder)
            if match:
                ent_type = match.group(1)
                if ent_type in ("PERSON", "GPE", "ORG"):
                    original = original.title()
            restored = re.sub(re.escape(placeholder), original, restored, flags=re.IGNORECASE)
        logger.debug(f"Restored Text: '{restored}'")
        return restored
