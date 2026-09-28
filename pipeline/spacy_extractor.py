"""spaCy-based entity and relationship extraction module.

Uses spaCy NER + dependency parsing to extract structured graph data
from text chunks WITHOUT calling an LLM.  This is the fast alternative
to the Ollama-based graph_extractor.py and returns the same
ExtractionResult dataclass so both are interchangeable.

Relationship strategies (in priority order):
  1. Dependency-parsed SVO triples  — grammatically grounded
  2. Sentence-level co-occurrence   — entities in the same sentence
     get a CO_OCCURS_WITH edge (only when no SVO triple found)

Noise reduction (pipeline/entity_filter.py):
  - EntityFilter     — drops purely-numeric, artifact, stopword entities
  - EntityNormalizer — collapses variant names (GPT4 / GPT-4 / GPT 4 → GPT-4)
  - RelationshipFilter — drops orphan edges, self-loops, weak-verb SVO triples
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── spaCy import with a helpful error message ─────────────────────────────────
try:
    import spacy
    from spacy.tokens import Doc, Span
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "spaCy is not installed. Run: pip install spacy && python -m spacy download en_core_web_sm"
    ) from exc

# Reuse the same dataclasses as the LLM extractor so both are interchangeable
from .graph_extractor import Entity, Relationship, ExtractionResult

# Noise-reduction layer (zero LLM calls)
from .entity_filter import EntityFilter, EntityNormalizer, RelationshipFilter

# Module-level cached filter instances — constructed once, reused per chunk
_entity_filter   = EntityFilter()
_entity_normalizer = EntityNormalizer()
_rel_filter      = RelationshipFilter()

# ── spaCy entity label → our canonical type names ────────────────────────────
LABEL_MAP: dict[str, str] = {
    "PERSON":      "Person",
    "ORG":         "Organization",
    "GPE":         "Location",
    "LOC":         "Location",
    "FAC":         "Facility",
    "PRODUCT":     "Product",
    "EVENT":       "Event",
    "WORK_OF_ART": "WorkOfArt",
    "LAW":         "Law",
    "LANGUAGE":    "Language",
    "DATE":        "Date",
    "TIME":        "Time",
    "MONEY":       "Money",
    "PERCENT":     "Percent",
    "QUANTITY":    "Quantity",
    "NORP":        "Group",          # nationalities, religious/political groups
}

# Labels we want to extract (filter out noise like CARDINAL, ORDINAL)
WANTED_LABELS: frozenset[str] = frozenset(LABEL_MAP.keys())

# Minimum character length for an entity name (avoids single letters)
MIN_ENTITY_LEN = 2

# Dependency relation labels that signal a subject
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}

# Dependency relation labels that signal an object
OBJECT_DEPS = {"dobj", "pobj", "iobj", "attr", "oprd", "dative"}

# Maximum co-occurrence edges per chunk (keeps graph tidy)
MAX_COOCCURRENCE_EDGES = 20


_nlp: Optional[object] = None  # cached spaCy model (lazy-loaded)


def _get_nlp():
    """Lazy-load the spaCy model once and cache it."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc
    return _nlp


def _clean_entity_name(text: str) -> str:
    """Normalize an entity surface form."""
    name = text.strip()
    # Remove leading/trailing punctuation but keep internal hyphens, apostrophes
    name = re.sub(r"^[^\w]+|[^\w]+$", "", name)
    # Collapse internal whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def _extract_svo_triples(doc: Doc, ent_names: set[str]) -> list[tuple[str, str, str]]:
    """
    Extract Subject-Verb-Object triples where at least one of subject/object
    is a named entity we already found.

    Returns list of (subject_text, verb_lemma_upper, object_text).
    """
    triples: list[tuple[str, str, str]] = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        # Find subject(s)
        subjects = [
            child for child in token.children if child.dep_ in SUBJECT_DEPS
        ]
        # Find object(s)
        objects = [
            child for child in token.children if child.dep_ in OBJECT_DEPS
        ]

        for subj in subjects:
            subj_span = _expand_noun_phrase(subj)
            subj_text = _clean_entity_name(subj_span)
            if not subj_text or len(subj_text) < MIN_ENTITY_LEN:
                continue

            for obj in objects:
                obj_span = _expand_noun_phrase(obj)
                obj_text = _clean_entity_name(obj_span)
                if not obj_text or len(obj_text) < MIN_ENTITY_LEN:
                    continue

                # At least one endpoint should be a known entity
                if subj_text in ent_names or obj_text in ent_names:
                    verb = token.lemma_.upper().replace(" ", "_")
                    verb = re.sub(r"[^A-Z0-9_]", "", verb) or "ASSOCIATED_WITH"
                    triples.append((subj_text, verb, obj_text))

    return triples


def _expand_noun_phrase(token) -> str:
    """Walk up to the noun chunk that contains this token, if available."""
    for chunk in token.doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            return chunk.text
    return token.text


def _extract_entities_from_doc(doc: Doc) -> list[Entity]:
    """Extract named entities from a processed spaCy Doc."""
    seen: set[str] = set()
    entities: list[Entity] = []

    for ent in doc.ents:
        if ent.label_ not in WANTED_LABELS:
            continue
        name = _clean_entity_name(ent.text)
        if len(name) < MIN_ENTITY_LEN:
            continue
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        entities.append(
            Entity(
                name=name,
                type=LABEL_MAP[ent.label_],
                description=f"Extracted by spaCy NER ({ent.label_})",
            )
        )

    return entities


def _extract_relationships_from_doc(
    doc: Doc,
    entities: list[Entity],
    source_text: str,
) -> list[Relationship]:
    """
    Extract relationships using:
      1. SVO dependency triples (preferred — grammatically grounded)
      2. Sentence-level co-occurrence (fallback)
    """
    ent_names: set[str] = {e.name for e in entities}
    ent_names_lower: dict[str, str] = {e.name.lower(): e.name for e in entities}
    relationships: list[Relationship] = []
    seen_rels: set[tuple[str, str, str]] = set()

    def _resolve(name: str) -> str:
        """Map a surface form to a canonical entity name if possible."""
        return ent_names_lower.get(name.lower(), name)

    # --- Strategy 1: SVO triples ---
    triples = _extract_svo_triples(doc, ent_names)
    for subj_text, verb, obj_text in triples:
        src = _resolve(subj_text)
        tgt = _resolve(obj_text)
        key = (src, tgt, verb)
        if key not in seen_rels:
            seen_rels.add(key)
            relationships.append(
                Relationship(
                    source=src,
                    target=tgt,
                    relationship=verb,
                    description=f"SVO triple extracted from: \"{source_text[:120]}\"",
                )
            )

    # --- Strategy 2: Sentence-level co-occurrence (entities in same sentence) ---
    co_count = 0
    for sent in doc.sents:
        # Collect entities that appear in this sentence
        sent_ents = [
            e for e in entities
            if e.name.lower() in sent.text.lower()
        ]
        if len(sent_ents) < 2:
            continue

        # Pair up entities — prefer connecting to the first entity in the sentence
        for i in range(len(sent_ents)):
            for j in range(i + 1, len(sent_ents)):
                if co_count >= MAX_COOCCURRENCE_EDGES:
                    break
                src = sent_ents[i].name
                tgt = sent_ents[j].name
                key = (src, tgt, "CO_OCCURS_WITH")
                rev_key = (tgt, src, "CO_OCCURS_WITH")
                # Skip if a directed SVO triple already covers this pair
                pair_covered = any(
                    (r.source == src and r.target == tgt) or
                    (r.source == tgt and r.target == src)
                    for r in relationships
                )
                if key not in seen_rels and rev_key not in seen_rels and not pair_covered:
                    seen_rels.add(key)
                    relationships.append(
                        Relationship(
                            source=src,
                            target=tgt,
                            relationship="CO_OCCURS_WITH",
                            description=f"Co-occur in sentence: \"{sent.text[:120]}\"",
                        )
                    )
                    co_count += 1

    return relationships


def extract_entities_and_relationships_spacy(
    chunks: list,
    document_id: str,
    entity_filter: Optional[EntityFilter] = None,
    entity_normalizer: Optional[EntityNormalizer] = None,
    rel_filter: Optional[RelationshipFilter] = None,
) -> list[ExtractionResult]:
    """Extract entities and relationships using spaCy NER + dependency parsing.

    Args:
        chunks: List of Chunk objects (same format as fed to Ollama extractor).
        document_id: Unique document identifier.
        entity_filter: Optional EntityFilter override (uses module-level default).
        entity_normalizer: Optional EntityNormalizer override.
        rel_filter: Optional RelationshipFilter override.

    Returns:
        List of ExtractionResult objects — identical schema to Ollama extractor output.

    Notes:
        - Zero LLM calls are made.
        - Processes all chunks in a single spaCy pipeline pass (batched via nlp.pipe).
        - Model is lazy-loaded and cached after first call.
        - Noise reduction (filter + normalize + rel-filter) applied per chunk.
    """
    nlp = _get_nlp()
    results: list[ExtractionResult] = []

    # Use provided overrides or fall back to module-level cached instances
    ent_filt = entity_filter   if entity_filter   is not None else _entity_filter
    ent_norm = entity_normalizer if entity_normalizer is not None else _entity_normalizer
    r_filt   = rel_filter      if rel_filter       is not None else _rel_filter

    # Batch process for efficiency — spaCy pipe is faster than calling nlp() per chunk
    texts = [chunk.text for chunk in chunks]

    for chunk, doc in zip(chunks, nlp.pipe(texts, batch_size=32)):
        # ── Raw extraction ────────────────────────────────────────────────
        raw_entities = _extract_entities_from_doc(doc)

        # ── Entity noise reduction ────────────────────────────────────────
        filtered_entities, _ = ent_filt.filter(raw_entities)
        entities = ent_norm.normalize(filtered_entities)

        # ── Relationship extraction (uses cleaned entity set) ─────────────
        raw_relationships = _extract_relationships_from_doc(
            doc, entities, source_text=chunk.text
        )

        # ── Relationship noise reduction ──────────────────────────────────
        entity_name_set = {e.name for e in entities}
        relationships, _ = r_filt.filter(raw_relationships, entity_names=entity_name_set)

        results.append(
            ExtractionResult(
                entities=entities,
                relationships=relationships,
                chunk_id=chunk.chunk_id,
                document_id=document_id,
            )
        )

    return results
