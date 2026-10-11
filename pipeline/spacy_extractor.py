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
from config.settings import ENABLE_COOCCURRENCE, MAX_COOCCURRENCE_EDGES
from .graph_extractor import Entity, Relationship, ExtractionResult

# Noise-reduction layer (zero LLM calls)
from .entity_filter import (
    EntityFilter,
    EntityNormalizer,
    RelationshipFilter,
    PredicateNormalizer,
)

# Module-level cached filter instances — constructed once, reused per chunk
_entity_filter   = EntityFilter()
_entity_normalizer = EntityNormalizer()
_pred_normalizer = PredicateNormalizer()
_rel_filter      = RelationshipFilter(predicate_normalizer=_pred_normalizer)

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


def _expand_noun_phrase(token) -> str:
    """Walk up to the noun chunk that contains this token, if available."""
    for ent in token.doc.ents:
        if ent.start <= token.i < ent.end:
            return ent.text
    for chunk in token.doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            return chunk.text
    return token.text


def _match_entity_span(token, doc: Doc, ent_names_lower: dict[str, str]) -> Optional[str]:
    """Find the entity name covering this token, or return cleaned phrase."""
    # 1. Named entity span (highest priority)
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            cleaned = _clean_entity_name(ent.text)
            return ent_names_lower.get(cleaned.lower(), cleaned)

    # 2. Noun chunk
    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            cleaned = _clean_entity_name(chunk.text)
            return ent_names_lower.get(cleaned.lower(), cleaned)

    # 3. Direct token
    cleaned = _clean_entity_name(token.text)
    if not cleaned or len(cleaned) < MIN_ENTITY_LEN:
        return None
    return ent_names_lower.get(cleaned.lower(), cleaned)


def _extract_svo_and_prepositional_triples(
    doc: Doc,
    ent_names: set[str],
    ent_names_lower: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    """Extract active SVO, passive voice agent, and prepositional dependency triples."""
    triples: list[tuple[str, str, str, str]] = []

    for token in doc:
        if token.pos_ not in ("VERB", "AUX"):
            continue

        active_subjs = [
            c for c in token.children
            if c.dep_ in SUBJECT_DEPS and c.dep_ not in ("nsubjpass", "csubjpass")
        ]
        passive_subjs = [
            c for c in token.children
            if c.dep_ in ("nsubjpass", "csubjpass")
        ]
        direct_objs = [
            c for c in token.children
            if c.dep_ in ("dobj", "iobj", "attr", "oprd")
        ]

        prep_attachments: list[tuple[str, object]] = []
        for child in token.children:
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        prep_attachments.append((child.text.upper(), pobj))

        verb_lemma = token.lemma_.upper().replace(" ", "_")
        verb_lemma = re.sub(r"[^A-Z0-9_]", "", verb_lemma) or "ASSOCIATED_WITH"

        # Case 1: Passive voice with "by" (e.g. "X was developed by Y")
        # Direct active flow: (Actor Y) --[DEVELOPED]--> (Target X)
        if passive_subjs:
            for p_subj in passive_subjs:
                patient_name = _match_entity_span(p_subj, doc, ent_names_lower)
                if not patient_name:
                    continue

                for prep_text, pobj_token in prep_attachments:
                    if prep_text == "BY":
                        actor_name = _match_entity_span(pobj_token, doc, ent_names_lower)
                        if actor_name and actor_name.lower() != patient_name.lower():
                            if actor_name.lower() in ent_names_lower or patient_name.lower() in ent_names_lower:
                                triples.append((
                                    actor_name,
                                    verb_lemma,
                                    patient_name,
                                    f"Passive voice triple: {actor_name} -> {verb_lemma} -> {patient_name}",
                                ))

        # Case 2: Active SVO with direct objects (e.g. "OpenAI produces GPT-4o")
        if active_subjs and direct_objs:
            for a_subj in active_subjs:
                subj_name = _match_entity_span(a_subj, doc, ent_names_lower)
                if not subj_name:
                    continue

                for d_obj in direct_objs:
                    obj_name = _match_entity_span(d_obj, doc, ent_names_lower)
                    if obj_name and obj_name.lower() != subj_name.lower():
                        if subj_name.lower() in ent_names_lower or obj_name.lower() in ent_names_lower:
                            triples.append((
                                subj_name,
                                verb_lemma,
                                obj_name,
                                f"Active SVO triple: {subj_name} -> {verb_lemma} -> {obj_name}",
                            ))

                    # Exemplifying prepositions on direct object (e.g. "produces models including GPT-4o")
                    for prep in d_obj.children:
                        if prep.dep_ == "prep" and prep.text.lower() in ("including", "such", "like"):
                            for pobj in prep.children:
                                if pobj.dep_ in ("pobj", "conj"):
                                    pobj_name = _match_entity_span(pobj, doc, ent_names_lower)
                                    if pobj_name and pobj_name.lower() != subj_name.lower():
                                        if subj_name.lower() in ent_names_lower or pobj_name.lower() in ent_names_lower:
                                            triples.append((
                                                subj_name,
                                                verb_lemma,
                                                pobj_name,
                                                f"SVO with exemplar: {subj_name} -> {verb_lemma} -> {pobj_name}",
                                            ))

        # Case 3: Active SVO with prepositional phrase (e.g. "Microsoft invested in OpenAI")
        if active_subjs and prep_attachments:
            for a_subj in active_subjs:
                subj_name = _match_entity_span(a_subj, doc, ent_names_lower)
                if not subj_name:
                    continue

                for prep_text, pobj_token in prep_attachments:
                    if prep_text in ("BY", "IN_ORDER_TO"):
                        continue
                    obj_name = _match_entity_span(pobj_token, doc, ent_names_lower)
                    if obj_name and obj_name.lower() != subj_name.lower():
                        if subj_name.lower() in ent_names_lower or obj_name.lower() in ent_names_lower:
                            rel_verb = f"{verb_lemma}_{prep_text}"
                            triples.append((
                                subj_name,
                                rel_verb,
                                obj_name,
                                f"Prepositional triple: {subj_name} -> {rel_verb} -> {obj_name}",
                            ))

    return triples


def _extract_adnominal_and_copular_triples(
    doc: Doc,
    ent_names_lower: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    """Extract adnominal participle clauses (e.g. 'database developed by Neo4j Inc') and copular/appositive relations."""
    triples: list[tuple[str, str, str, str]] = []

    for token in doc:
        # Copular sentences: "X is a Y developed by Z" or "X is the CEO of Y"
        if token.lemma_ == "be" and token.pos_ in ("AUX", "VERB"):
            subjs = [c for c in token.children if c.dep_ in ("nsubj", "csubj")]
            attrs = [c for c in token.children if c.dep_ in ("attr", "acomp")]

            for subj in subjs:
                subj_name = _match_entity_span(subj, doc, ent_names_lower)
                if not subj_name:
                    continue

                for attr in attrs:
                    # Relational noun (e.g. "Demis Hassabis is the CEO of Google DeepMind")
                    for prep in attr.children:
                        if prep.dep_ == "prep" and prep.text.lower() in ("of", "for", "at", "in"):
                            for pobj in prep.children:
                                if pobj.dep_ == "pobj":
                                    pobj_name = _match_entity_span(pobj, doc, ent_names_lower)
                                    if pobj_name and pobj_name.lower() != subj_name.lower():
                                        if subj_name.lower() in ent_names_lower or pobj_name.lower() in ent_names_lower:
                                            pred = f"{attr.lemma_.upper()}_{prep.text.upper()}"
                                            triples.append((
                                                subj_name,
                                                pred,
                                                pobj_name,
                                                f"Copular relational noun: {subj_name} -> {pred} -> {pobj_name}",
                                            ))

                    # Adnominal participle/clause (e.g. "database developed by Neo4j Inc")
                    for clause in attr.children:
                        if clause.dep_ in ("acl", "relcl") and clause.pos_ == "VERB":
                            verb_lemma = clause.lemma_.upper()
                            for prep in clause.children:
                                if prep.dep_ in ("prep", "agent") and prep.text.lower() == "by":
                                    pobjs = [c for c in prep.children if c.dep_ in ("pobj", "dative")]
                                    if not pobjs and prep.i + 1 < len(doc):
                                        pobjs = [doc[prep.i + 1]]
                                    for pobj in pobjs:
                                        actor_name = _match_entity_span(pobj, doc, ent_names_lower)
                                        if actor_name and actor_name.lower() != subj_name.lower():
                                            if actor_name.lower() in ent_names_lower or subj_name.lower() in ent_names_lower:
                                                triples.append((
                                                    actor_name,
                                                    verb_lemma,
                                                    subj_name,
                                                    f"Adnominal clause: {actor_name} -> {verb_lemma} -> {subj_name}",
                                                ))

        # Appositive relational nouns (e.g. "Demis Hassabis, CEO of Google DeepMind")
        if token.dep_ == "appos":
            head_name = _match_entity_span(token.head, doc, ent_names_lower)
            if head_name:
                for prep in token.children:
                    if prep.dep_ == "prep" and prep.text.lower() in ("of", "at", "for"):
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                pobj_name = _match_entity_span(pobj, doc, ent_names_lower)
                                if pobj_name and pobj_name.lower() != head_name.lower():
                                    if head_name.lower() in ent_names_lower or pobj_name.lower() in ent_names_lower:
                                        pred = f"{token.lemma_.upper()}_{prep.text.upper()}"
                                        triples.append((
                                            head_name,
                                            pred,
                                            pobj_name,
                                            f"Appositive relational noun: {head_name} -> {pred} -> {pobj_name}",
                                        ))

    return triples


def _extract_entities_from_doc(doc: Doc) -> list[Entity]:
    """Extract named entities from a processed spaCy Doc."""
    seen: set[str] = set()
    entities: list[Entity] = []

    # 1. spaCy Named Entities
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

    # 2. Extract Proper Nouns (PROPN) from noun chunks that small NER models miss
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PROPN":
            name = _clean_entity_name(chunk.text)
            if len(name) < MIN_ENTITY_LEN:
                continue
            if name.lower() in seen:
                continue
            if not any(c.isupper() for c in name):
                continue
            seen.add(name.lower())
            entities.append(
                Entity(
                    name=name,
                    type="Technology",
                    description="Extracted as Proper Noun phrase",
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
      1. Active SVO, passive voice, and prepositional dependency triples
      2. Adnominal participle and copular relational noun triples
      3. Gated sentence-level co-occurrence (only if ENABLE_COOCCURRENCE is True)
    """
    ent_names: set[str] = {e.name for e in entities}
    ent_names_lower: dict[str, str] = {e.name.lower(): e.name for e in entities}
    relationships: list[Relationship] = []
    seen_rels: set[tuple[str, str, str]] = set()

    def _resolve(name: str) -> str:
        return ent_names_lower.get(name.lower(), name)

    # --- Strategy 1: SVO & Prepositional Triples ---
    svo_triples = _extract_svo_and_prepositional_triples(doc, ent_names, ent_names_lower)
    for src_raw, verb, tgt_raw, desc in svo_triples:
        src = _resolve(src_raw)
        tgt = _resolve(tgt_raw)
        key = (src, tgt, verb)
        if key not in seen_rels and src != tgt:
            seen_rels.add(key)
            relationships.append(
                Relationship(
                    source=src,
                    target=tgt,
                    relationship=verb,
                    description=desc,
                )
            )

    # --- Strategy 2: Adnominal & Copular Triples ---
    adn_triples = _extract_adnominal_and_copular_triples(doc, ent_names_lower)
    for src_raw, verb, tgt_raw, desc in adn_triples:
        src = _resolve(src_raw)
        tgt = _resolve(tgt_raw)
        key = (src, tgt, verb)
        if key not in seen_rels and src != tgt:
            seen_rels.add(key)
            relationships.append(
                Relationship(
                    source=src,
                    target=tgt,
                    relationship=verb,
                    description=desc,
                )
            )

    # --- Strategy 3: Gated Sentence-level co-occurrence (strictly gated fallback) ---
    if ENABLE_COOCCURRENCE:
        co_count = 0
        for sent in doc.sents:
            sent_ents = [
                e for e in entities
                if e.name.lower() in sent.text.lower()
            ]
            if len(sent_ents) < 2:
                continue

            for i in range(len(sent_ents)):
                for j in range(i + 1, len(sent_ents)):
                    if co_count >= MAX_COOCCURRENCE_EDGES:
                        break
                    src = sent_ents[i].name
                    tgt = sent_ents[j].name
                    key = (src, tgt, "CO_OCCURS_WITH")
                    rev_key = (tgt, src, "CO_OCCURS_WITH")
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
