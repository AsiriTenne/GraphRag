"""Entity and relationship noise-reduction filters for the graph extraction pipeline.

Provides three composable, configurable classes:

  EntityFilter     — drops purely-numeric, too-short, stopword, and PDF-artifact entities
  EntityNormalizer — collapses tech-name variant families (GPT4/GPT-4/GPT 4 → GPT-4)
  RelationshipFilter — drops orphan edges, self-loops, and weak-verb SVO triples

None of these classes make LLM calls. They operate purely on text patterns and sets.
All thresholds and word-lists are overridable via constructor kwargs so you can tune
behaviour from config/settings.py without editing pipeline code.

Usage (within spacy_extractor.py):
    from .entity_filter import EntityFilter, EntityNormalizer, RelationshipFilter
    filt   = EntityFilter()
    norm   = EntityNormalizer()
    rfilt  = RelationshipFilter()

    entities      = filt.filter(entities)
    entities      = norm.normalize(entities)
    relationships = rfilt.filter(relationships, entity_names={e.name for e in entities})
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

from .graph_extractor import Entity, Relationship


# ─────────────────────────────────────────────────────────────────────────────
# Default configuration constants (override via constructor kwargs)
# ─────────────────────────────────────────────────────────────────────────────

#: Entity types that are allowed to be filtered as purely-numeric tokens.
#: PERSON, ORG, GPE, PRODUCT etc. are never purely-numeric so they're safe.
_NUMERIC_FILTERABLE_TYPES: frozenset[str] = frozenset({
    "Date", "Time", "Money", "Percent", "Quantity",
})

#: Entity types that are always preserved (bypass most filters).
_ALWAYS_KEEP_TYPES: frozenset[str] = frozenset({
    "Person", "Organization", "Location", "Product", "Facility",
    "WorkOfArt", "Law", "Language", "Group", "Event",
})

#: Minimum character length for an entity name after cleaning.
_MIN_NAME_LEN: int = 3

#: Generic structural/functional words that add no graph value.
_STOPWORD_ENTITIES: frozenset[str] = frozenset({
    # document structure
    "figure", "table", "section", "appendix", "chapter", "page",
    "exhibit", "panel", "box", "note", "footnote", "header", "footer",
    "et al", "et al.", "ibid", "ibid.",
    # generic ML/NLP words that spaCy often tags as ORG/PRODUCT
    "model", "models", "system", "systems", "dataset", "datasets",
    "baseline", "baselines", "benchmark", "benchmarks",
    "result", "results", "output", "outputs", "input", "inputs",
    "task", "tasks", "prompt", "prompts", "response", "responses",
    "text", "texts", "data", "sample", "samples",
    "training", "testing", "evaluation", "inference",
    "example", "examples", "method", "methods", "approach",
    "paper", "report", "study", "work", "analysis",
    "human", "humans", "user", "users",
    "ai", "ml", "nlp",  # too generic as standalone names
    # Roman numerals / ordinals that slip through
    "i", "ii", "iii", "iv", "vi", "vii", "viii", "ix",
})

#: Regex patterns that identify PDF extraction artifacts.
_ARTIFACT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*$"),
    re.compile(r"^[\W_]+$"),
    re.compile(r"^\d+[\.\)]\s*$"),
    re.compile(r"^[A-Z]\.$"),
    re.compile(r"\n"),
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^\d{4}[\u2013\-]\d{2,4}$"),
    re.compile(r"^\d+[%\$\xa3\u20ac]$"),
    re.compile(r"^[A-Z0-9]{1,3}$"),
    re.compile(r"^[A-Za-z]{1,3}$"),
    re.compile(r"^[\u2022\u25cf\u2013\u2014]+"),
    re.compile(r"\.{2,}"),
    re.compile(r"^(table|figure|fig|tab|eq|equation|algorithm|alg)\s*\d+", re.IGNORECASE),
    re.compile(r"^(section|sec|chapter|ch|appendix|app)\s*[\d\.]+", re.IGNORECASE),
    re.compile(r"^[a-z]{1,2}$"),
    re.compile(r"^\W"),
    # PDF run-together tokens: >40 chars, no spaces
    re.compile(r"^\S{41,}$"),
    # Citation bracket artifacts: PaLM[3, QDGAT[59]
    re.compile(r"\[\d"),
    # Benchmark score rows: "36/60 38/60", "100% 80% 60%"
    re.compile(r"^\d+[/%]\s*\d+[/%]"),
    re.compile(r"\d+%\s+\d+%\s+\d+%"),
    # Consonant chains: broken OCR (no vowels, 6+ chars)
    re.compile(r"^[^aeiouAEIOU\s]{6,}$"),
    # Exam score notation: "5(86th-100th)"
    re.compile(r"^\d+\(\d+"),
)

#: SVO verbs (uppercase lemmas) too weak to form a meaningful graph edge.
_WEAK_VERBS: frozenset[str] = frozenset({
    "BE", "IS", "ARE", "WAS", "WERE", "BEEN",
    "HAVE", "HAS", "HAD",
    "DO", "DOES", "DID",
    "GET", "GOT", "GOTTEN",
    "MAKE", "MADE",
    "TAKE", "TOOK",
    "GO", "WENT",
    "COME", "CAME",
    "GIVE", "GAVE",
    "SAY", "SAID",
    "KNOW", "KNEW",
    "SEE", "SAW",
    "SEEM", "SEEMED",
    "BECOME", "BECAME",
    "APPEAR", "APPEARED",
    "REMAIN", "REMAINED",
    "CONTAIN", "CONTAINS",
    "INCLUDE", "INCLUDES",
    "SHOW", "SHOWS", "SHOWED",
    "FIND", "FINDS", "FOUND",
    "NOTE", "NOTES", "NOTED",
    "DISCUSS", "DISCUSSES",
    "DESCRIBE", "DESCRIBES",
    "PRESENT", "PRESENTS",
    "REPORT", "REPORTS",
    "MENTION", "MENTIONS",
    "INDICATE", "INDICATES",
    "SUGGEST", "SUGGESTS",
    "PROVIDE", "PROVIDES",
    "FOLLOW", "FOLLOWS",
    "COMPARE", "COMPARES",
})

# ─────────────────────────────────────────────────────────────────────────────
# Tech-name normalization tables
# ─────────────────────────────────────────────────────────────────────────────

#: (compiled_pattern, canonical_name) pairs applied in order.
_TECH_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    # GPT family — most specific first
    (re.compile(r"\bGPT[\s\-_]?4o\b", re.IGNORECASE),    "GPT-4o"),
    (re.compile(r"\bGPT[\s\-_]?4\b", re.IGNORECASE),      "GPT-4"),
    (re.compile(r"\bGPT[\s\-_]?3\.?5\b", re.IGNORECASE),  "GPT-3.5"),
    (re.compile(r"\bGPT[\s\-_]?3\b", re.IGNORECASE),      "GPT-3"),
    # LLaMA family  — match LLaMA, Llama, llama with optional space/hyphen before digit
    (re.compile(r'\b[Ll][Ll][aA][mM][aA][\s\-_]?2\b'),  "LLaMA-2"),
    (re.compile(r'\b[Ll][Ll][aA][mM][aA][\s\-_]?3\b'),  "LLaMA-3"),
    (re.compile(r'\b[Ll][Ll][aA][mM][aA]\b'),             "LLaMA"),
    # Claude
    (re.compile(r"\bClaude[\s\-_]?3[\s\-_]?[Oo]pus\b",   re.IGNORECASE), "Claude 3 Opus"),
    (re.compile(r"\bClaude[\s\-_]?3[\s\-_]?[Ss]onnet\b", re.IGNORECASE), "Claude 3 Sonnet"),
    (re.compile(r"\bClaude[\s\-_]?3[\s\-_]?[Hh]aiku\b",  re.IGNORECASE), "Claude 3 Haiku"),
    (re.compile(r"\bClaude[\s\-_]?3\b",   re.IGNORECASE),  "Claude 3"),
    (re.compile(r"\bClaude[\s\-_]?2\b",   re.IGNORECASE),  "Claude 2"),
    # BERT family
    (re.compile(r"\bBERT[\s\-_]?[Bb]ase\b"),   "BERT-base"),
    (re.compile(r"\bBERT[\s\-_]?[Ll]arge\b"),  "BERT-large"),
    (re.compile(r"\bRoBERTa\b", re.IGNORECASE), "RoBERTa"),
    # Gemini
    (re.compile(r"\bGemini[\s\-_]?Ultra\b",  re.IGNORECASE), "Gemini Ultra"),
    (re.compile(r"\bGemini[\s\-_]?Pro\b",    re.IGNORECASE), "Gemini Pro"),
    (re.compile(r"\bGemini[\s\-_]?Nano\b",   re.IGNORECASE), "Gemini Nano"),
    # PaLM
    (re.compile(r"\bPaLM[\s\-_]?2\b", re.IGNORECASE), "PaLM 2"),
    (re.compile(r"\bPaLM\b",           re.IGNORECASE), "PaLM"),
    # Falcon
    (re.compile(r"\bFalcon[\s\-_]?40[Bb]\b", re.IGNORECASE), "Falcon-40B"),
    (re.compile(r"\bFalcon[\s\-_]?7[Bb]\b",  re.IGNORECASE), "Falcon-7B"),
]

#: lowercase surface form → canonical form.
_KNOWN_ALIASES: dict[str, str] = {
    "openai":          "OpenAI",
    "open ai":         "OpenAI",
    "open-ai":         "OpenAI",
    "deepmind":        "DeepMind",
    "deep mind":       "DeepMind",
    "google deepmind": "Google DeepMind",
    "anthropic":       "Anthropic",
    "meta ai":         "Meta AI",
    "meta-ai":         "Meta AI",
    "hugging face":    "Hugging Face",
    "huggingface":     "Hugging Face",
    "microsoft":       "Microsoft",
    "bing chat":       "Bing Chat",
    "chatgpt":         "ChatGPT",
    "chat gpt":        "ChatGPT",
    "gpt4":            "GPT-4",
    "gpt3":            "GPT-3",
    "gpt 4":           "GPT-4",
    "gpt 3":           "GPT-3",
}


# ─────────────────────────────────────────────────────────────────────────────
# EntityFilter
# ─────────────────────────────────────────────────────────────────────────────

class EntityFilter:
    """Filter noisy entities extracted from PDF text.

    Removes:
    - purely-numeric or symbol-only names
    - names shorter than min_name_len
    - names in the stopword set
    - names matching PDF artifact patterns
    - numeric-type entities (Date/Time/Money/…) that carry only raw numbers

    Args:
        min_name_len: Minimum character length (default 3).
        stopwords: Replace the default stopword set. Use extra_stopwords to extend.
        artifact_patterns: Replace the default artifact pattern tuple.
        numeric_filterable_types: Entity types subject to the numeric-only check.
        always_keep_types: Entity types that bypass most filters.
        extra_stopwords: Additional stopwords merged with the defaults.
    """

    def __init__(
        self,
        min_name_len: int = _MIN_NAME_LEN,
        max_name_len: int = 60,
        stopwords: Optional[frozenset[str]] = None,
        artifact_patterns: Optional[tuple] = None,
        numeric_filterable_types: Optional[frozenset[str]] = None,
        always_keep_types: Optional[frozenset[str]] = None,
        extra_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.min_name_len = min_name_len
        self.max_name_len = max_name_len
        self._stopwords: frozenset[str] = (
            stopwords if stopwords is not None else _STOPWORD_ENTITIES
        )
        if extra_stopwords:
            self._stopwords = self._stopwords | frozenset(s.lower() for s in extra_stopwords)
        self._artifact_patterns = (
            artifact_patterns if artifact_patterns is not None else _ARTIFACT_PATTERNS
        )
        self._numeric_filterable_types = (
            numeric_filterable_types
            if numeric_filterable_types is not None
            else _NUMERIC_FILTERABLE_TYPES
        )
        self._always_keep_types = (
            always_keep_types
            if always_keep_types is not None
            else _ALWAYS_KEEP_TYPES
        )

    def _is_purely_numeric(self, name: str) -> bool:
        stripped = re.sub(r"[\$£€¥₹%,\.\s]", "", name)
        return stripped.isdigit() or stripped == ""

    def _has_artifact_pattern(self, name: str) -> bool:
        return any(p.search(name) for p in self._artifact_patterns)

    def _has_control_chars(self, name: str) -> bool:
        return any(unicodedata.category(ch).startswith("C") for ch in name)

    def _is_runtogether(self, name: str) -> bool:
        """Detect PDF run-together tokens (words concatenated without spaces).

        Three heuristics applied:
        1. Single long word (>12 chars, no spaces) without separators → concatenated.
        2. Multi-word with high avg word length (>10) → concatenated fragments.
        3. Spaced-out individual characters like 'a l P W o rld' → OCR artifact.
        """
        words = name.split()
        num_words = len(words)
        if not words:
            return True

        avg_word_len = sum(len(w) for w in words) / num_words

        # Heuristic 3: spaced-out individual characters (PDF column/OCR artifact)
        if num_words >= 4 and avg_word_len <= 2.0:
            return True

        # Heuristic 1: single word, long, no separator (hyphen/dot/slash)
        if num_words == 1 and len(name) > 12:
            has_separator = bool(re.search(r"[-./]", name))
            is_all_caps   = name.isupper()
            if not has_separator and not is_all_caps:
                return True

        # Heuristic 2: multi-word with high avg word length
        if len(name) > 20 and avg_word_len > 10:
            return True

        return False

    def should_keep(self, entity: Entity) -> bool:
        """Return True if entity should be kept in the graph."""
        name = entity.name.strip()

        # Hard length guards — applied to ALL types (no exception)
        if len(name) < self.min_name_len:
            return False
        if len(name) > self.max_name_len:
            return False

        # Run-together token guard — applied to ALL types
        if self._is_runtogether(name):
            return False

        # Always-keep types: apply remaining hard guards then pass through
        if entity.type in self._always_keep_types:
            if name.lower() in self._stopwords:
                return False
            if self._has_artifact_pattern(name):
                return False
            return True

        # Length guard
        if len(name) < self.min_name_len:
            return False

        # Stopword guard
        if name.lower() in self._stopwords:
            return False

        # Control character / mojibake guard
        if self._has_control_chars(name):
            return False

        # Artifact pattern guard
        if self._has_artifact_pattern(name):
            return False

        # Purely-numeric guard for numeric-type entities
        if entity.type in self._numeric_filterable_types:
            if self._is_purely_numeric(name):
                return False

        return True

    def filter(
        self, entities: list[Entity]
    ) -> tuple[list[Entity], list[Entity]]:
        """Return (kept_entities, removed_entities)."""
        kept, removed = [], []
        for ent in entities:
            (kept if self.should_keep(ent) else removed).append(ent)
        return kept, removed


# ─────────────────────────────────────────────────────────────────────────────
# EntityNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class EntityNormalizer:
    """Normalize entity names to canonical forms.

    Applies in order:
    1. Tech-name regex patterns (GPT4 → GPT-4, LLaMA 2 → LLaMA-2, …)
    2. Known-alias exact map (openai → OpenAI, gpt 4 → GPT-4, …)

    After normalization, duplicates that collapse to the same name are
    deduplicated (first occurrence wins).

    Args:
        tech_patterns: Override the default tech-name pattern list.
        known_aliases: Override the default alias dict.
        extra_aliases: Additional lowercase→canonical pairs merged with defaults.
    """

    def __init__(
        self,
        tech_patterns: Optional[list] = None,
        known_aliases: Optional[dict] = None,
        extra_aliases: Optional[dict] = None,
    ) -> None:
        self._tech_patterns = (
            tech_patterns if tech_patterns is not None else _TECH_NAME_PATTERNS
        )
        self._aliases: dict[str, str] = (
            dict(known_aliases) if known_aliases is not None else dict(_KNOWN_ALIASES)
        )
        if extra_aliases:
            self._aliases.update({k.lower(): v for k, v in extra_aliases.items()})

    def normalize_name(self, name: str) -> str:
        """Return the canonical form of name."""
        result = name.strip()
        for pattern, canonical in self._tech_patterns:
            if pattern.fullmatch(result):
                result = canonical
                break
            result = pattern.sub(canonical, result)
        alias_key = result.strip().lower()
        if alias_key in self._aliases:
            result = self._aliases[alias_key]
        return result.strip()

    def normalize(self, entities: list[Entity]) -> list[Entity]:
        """Normalize names and deduplicate collapsed variants.

        Returns:
            Deduplicated list with normalized names.
        """
        seen: dict[str, Entity] = {}
        result: list[Entity] = []
        for ent in entities:
            canonical = self.normalize_name(ent.name)
            key = canonical.lower()
            if key not in seen:
                new_ent = Entity(
                    name=canonical,
                    type=ent.type,
                    description=ent.description,
                )
                seen[key] = new_ent
                result.append(new_ent)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# RelationshipFilter
# ─────────────────────────────────────────────────────────────────────────────

class RelationshipFilter:
    """Filter noisy relationships.

    Removes:
    - Self-loops (source == target after normalization)
    - Orphan edges (endpoint not in the post-filter entity name set)
    - SVO triples with semantically-weak verbs

    CO_OCCURS_WITH edges bypass the weak-verb guard.

    Args:
        weak_verbs: Replace the default weak-verb set.
        extra_weak_verbs: Additional weak verbs merged with defaults.
        filter_orphans: If True (default), drop edges with non-entity endpoints.
    """

    def __init__(
        self,
        weak_verbs: Optional[frozenset[str]] = None,
        extra_weak_verbs: Optional[Iterable[str]] = None,
        filter_orphans: bool = True,
    ) -> None:
        self._weak_verbs: frozenset[str] = (
            weak_verbs if weak_verbs is not None else _WEAK_VERBS
        )
        if extra_weak_verbs:
            self._weak_verbs = self._weak_verbs | frozenset(
                v.upper() for v in extra_weak_verbs
            )
        self.filter_orphans = filter_orphans

    def should_keep(
        self,
        rel: Relationship,
        entity_names_lower: Optional[frozenset[str]] = None,
    ) -> bool:
        """Return True if rel should be kept."""
        src = rel.source.strip()
        tgt = rel.target.strip()

        # Self-loop guard
        if src.lower() == tgt.lower():
            return False

        # Orphan-endpoint guard
        if self.filter_orphans and entity_names_lower is not None:
            if src.lower() not in entity_names_lower:
                return False
            if tgt.lower() not in entity_names_lower:
                return False

        # Weak-verb guard (skip for CO_OCCURS_WITH)
        verb = rel.relationship.upper()
        if verb != "CO_OCCURS_WITH" and verb in self._weak_verbs:
            return False

        return True

    def filter(
        self,
        relationships: list[Relationship],
        entity_names: Optional[set[str]] = None,
    ) -> tuple[list[Relationship], list[Relationship]]:
        """Return (kept_relationships, removed_relationships).

        Args:
            relationships: Relationships to filter.
            entity_names: Canonical entity names (post-filter). Pass None to
                          skip the orphan-endpoint guard.
        """
        entity_names_lower: Optional[frozenset[str]] = (
            frozenset(n.lower() for n in entity_names) if entity_names else None
        )
        kept, removed = [], []
        for rel in relationships:
            (kept if self.should_keep(rel, entity_names_lower) else removed).append(rel)
        return kept, removed


# ─────────────────────────────────────────────────────────────────────────────
# Cross-chunk normalization helper (used by entity_resolver.py)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_entity_name_for_resolution(name: str) -> str:
    """Normalize a name for cross-chunk entity resolution.

    Applies the same tech-pattern and alias tables as EntityNormalizer so
    that GPT-4 and GPT4 from different chunks collapse to the same canonical
    node in entity_resolver.py.
    """
    result = name.strip()
    for pattern, canonical in _TECH_NAME_PATTERNS:
        if pattern.fullmatch(result):
            result = canonical
            break
        result = pattern.sub(canonical, result)
    alias_key = result.strip().lower()
    if alias_key in _KNOWN_ALIASES:
        result = _KNOWN_ALIASES[alias_key]
    return result.strip()
