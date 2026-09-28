"""Entity resolution module.

Normalizes and deduplicates entities extracted across multiple chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


from .graph_extractor import Entity, Relationship, ExtractionResult
from .entity_filter import normalize_entity_name_for_resolution


def _normalize_name(name: str) -> str:
    """Normalize an entity name for comparison.

    - Lowercase
    - Strip whitespace
    - Remove extra spaces
    - Remove trailing punctuation
    """
    normalized = name.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(".,;:!?")
    return normalized


@dataclass
class ResolvedEntity:
    """A resolved (deduplicated) entity."""
    canonical_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""


def resolve_entities(
    extraction_results: list[ExtractionResult],
) -> tuple[list[ResolvedEntity], list[dict], list[dict]]:
    """Resolve and deduplicate entities across all extraction results.

    Returns a tuple of:
        - List of ResolvedEntity objects (unique entities)
        - List of relationship dicts with resolved entity names
        - List of chunk-mention dicts linking chunk_id to canonical entity_name
    """
    # Map normalized name -> ResolvedEntity
    entity_map: dict[str, ResolvedEntity] = {}

    # First pass: collect all entities and find canonical names
    for result in extraction_results:
        for entity in result.entities:
            # Apply tech-name normalization first (GPT4 → GPT-4, openai → OpenAI, …)
            # so that variants from different chunks collapse to the same key.
            tech_normalized_name = normalize_entity_name_for_resolution(entity.name)
            normalized = _normalize_name(tech_normalized_name)

            if normalized in entity_map:
                existing = entity_map[normalized]
                # Keep the longer name as canonical
                if len(tech_normalized_name) > len(existing.canonical_name):
                    existing.aliases.append(existing.canonical_name)
                    existing.canonical_name = tech_normalized_name
                elif (
                    tech_normalized_name != existing.canonical_name
                    and tech_normalized_name not in existing.aliases
                ):
                    existing.aliases.append(tech_normalized_name)

                # Update description if we have a new one
                if entity.description and not existing.description:
                    existing.description = entity.description
            else:
                entity_map[normalized] = ResolvedEntity(
                    canonical_name=tech_normalized_name,
                    entity_type=entity.type,
                    description=entity.description,
                )

    # Build a normalized-name -> canonical-name lookup
    canonical_lookup: dict[str, str] = {
        norm: resolved.canonical_name
        for norm, resolved in entity_map.items()
    }

    # Second pass: resolve relationships
    resolved_relationships: list[dict] = []
    seen_rels: set[tuple[str, str, str]] = set()

    for result in extraction_results:
        for rel in result.relationships:
            source_norm = _normalize_name(rel.source)
            target_norm = _normalize_name(rel.target)

            source_canonical = canonical_lookup.get(source_norm, rel.source)
            target_canonical = canonical_lookup.get(target_norm, rel.target)

            rel_key = (source_canonical, target_canonical, rel.relationship)

            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                resolved_relationships.append(
                    {
                        "source": source_canonical,
                        "target": target_canonical,
                        "relationship": rel.relationship,
                        "description": rel.description,
                        "document_id": result.document_id,
                        "chunk_id": result.chunk_id,
                    }
                )

    # Third pass: resolve all chunk-entity mentions (for ALL entities extracted in chunks)
    chunk_mentions: list[dict] = []
    seen_mentions: set[tuple[str, str]] = set()

    for result in extraction_results:
        # Check direct entities extracted from chunk
        for entity in result.entities:
            norm = _normalize_name(entity.name)
            canonical = canonical_lookup.get(norm, entity.name)
            mention_key = (result.chunk_id, canonical)

            if mention_key not in seen_mentions:
                seen_mentions.add(mention_key)
                chunk_mentions.append(
                    {
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "entity_name": canonical,
                    }
                )

        # Also check relationship sources/targets to ensure no mention is missed
        for rel in result.relationships:
            for name in [rel.source, rel.target]:
                norm = _normalize_name(name)
                canonical = canonical_lookup.get(norm, name)
                mention_key = (result.chunk_id, canonical)

                if mention_key not in seen_mentions:
                    seen_mentions.add(mention_key)
                    chunk_mentions.append(
                        {
                            "chunk_id": result.chunk_id,
                            "document_id": result.document_id,
                            "entity_name": canonical,
                        }
                    )

    resolved_entities = list(entity_map.values())
    return resolved_entities, resolved_relationships, chunk_mentions

