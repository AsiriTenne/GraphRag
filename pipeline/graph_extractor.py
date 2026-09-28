"""Ollama LLM-based entity and relationship extraction module.

Sends text chunks to a local Ollama model (e.g. llama3.2) via its REST API
and receives structured entities and relationships.
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass

from config.settings import OLLAMA_HOST, OLLAMA_MODEL


@dataclass
class Entity:
    """An extracted entity."""
    name: str
    type: str
    description: str = ""


@dataclass
class Relationship:
    """An extracted relationship between two entities."""
    source: str
    target: str
    relationship: str
    description: str = ""


@dataclass
class ExtractionResult:
    """Result of entity/relationship extraction from one chunk."""
    entities: list[Entity]
    relationships: list[Relationship]
    chunk_id: str
    document_id: str


EXTRACTION_PROMPT = """You are an information extraction system. Analyze the following text and extract entities and relationships.

RULES FOR ENTITIES:
1. Extract meaningful domain entities (e.g., Organizations, Technologies, Concepts, Persons, Products, Frameworks).
2. Entity types should be descriptive (e.g., Organization, Technology, Concept, QueryLanguage, Framework, Model).
3. Do NOT extract system structural terms as domain entities (skip 'Document', 'Chunk', 'Entity', 'Node', 'Edge', 'Pipeline', 'Mentions').

RULES FOR RELATIONSHIPS & DIRECTION:
4. Relationships MUST follow active subject-predicate-object direction: (Subject/Actor) --[ACTIVE_RELATIONSHIP]--> (Object/Target).
   - Example: "Neo4j Inc." --[DEVELOPED]--> "Neo4j"
   - Example: "Neo4j" --[USES]--> "Cypher"
   - Example: "OpenAI" --[PRODUCES]--> "GPT-4o"
   - Example: "Google DeepMind" --[DEVELOPED]--> "Antigravity"
5. Do NOT use passive relationship types (DO NOT use DEVELOPED_BY, CREATED_BY, USED_BY, PRODUCED_BY). Use ACTIVE verbs: DEVELOPED, CREATED, USES, PRODUCES, INTEGRATES, HAS, ENHANCES.
6. Only extract relationships that are EXPLICITLY supported by the text. Do NOT connect entities across unrelated statements or infer unstated facts.
7. Each relationship must connect two extracted entities.

TEXT:
{text}

Return a JSON object with exactly this structure:
{{
  "entities": [
    {{"name": "Entity Name", "type": "EntityType", "description": "Brief description"}}
  ],
  "relationships": [
    {{"source": "Subject/Actor Name", "target": "Object/Target Name", "relationship": "ACTIVE_RELATIONSHIP", "description": "Brief description"}}
  ]
}}

Return ONLY valid JSON. No markdown, no explanation."""



def check_ollama_connection() -> tuple[bool, str]:
    """Check if the local Ollama server is running and accessible.

    Returns:
        (is_connected, message)
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GraphRAG/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return True, f"Ollama host online at {OLLAMA_HOST}. Available models: {models}"
            return False, f"Ollama returned HTTP status {response.status}"
    except Exception as e:
        return False, f"Failed to connect to Ollama at {OLLAMA_HOST}: {str(e)}"


def _parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response, handling potential markdown code blocks or trailing text."""
    text = response_text.strip()

    # Extract JSON block if wrapped in markdown
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # Find raw JSON object bounds
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]

    return json.loads(text)


def extract_entities_and_relationships(
    chunks: list,
    document_id: str,
) -> list[ExtractionResult]:
    """Extract entities and relationships from text chunks using local Ollama LLM.

    Args:
        chunks: List of Chunk objects from the chunker.
        document_id: The unique document identifier.

    Returns:
        List of ExtractionResult objects.

    Raises:
        ConnectionError: If Ollama server is unreachable.
        RuntimeError: If Ollama chat completion fails.
    """
    is_connected, msg = check_ollama_connection()
    if not is_connected:
        raise ConnectionError(f"Cannot extract entities. Ollama server is offline: {msg}")

    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    results: list[ExtractionResult] = []

    for chunk in chunks:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are an information extraction system. Return ONLY valid JSON."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(text=chunk.text)},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "GraphRAG/1.0"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama API returned status {response.status}")

                resp_body = response.read().decode("utf-8")
                resp_json = json.loads(resp_body)
                raw_content = resp_json.get("message", {}).get("content", "")

                parsed = _parse_llm_response(raw_content)

                entities = [
                    Entity(
                        name=e["name"],
                        type=e.get("type", "Unknown"),
                        description=e.get("description", ""),
                    )
                    for e in parsed.get("entities", [])
                    if isinstance(e, dict) and e.get("name")
                ]

                relationships = [
                    Relationship(
                        source=r["source"],
                        target=r["target"],
                        relationship=r.get("relationship", "ASSOCIATED_WITH"),
                        description=r.get("description", ""),
                    )
                    for r in parsed.get("relationships", [])
                    if isinstance(r, dict) and r.get("source") and r.get("target")
                ]

                results.append(
                    ExtractionResult(
                        entities=entities,
                        relationships=relationships,
                        chunk_id=chunk.chunk_id,
                        document_id=document_id,
                    )
                )

        except Exception as e:
            raise RuntimeError(
                f"Ollama LLM extraction failed for chunk {chunk.chunk_index} using model '{OLLAMA_MODEL}': {e}"
            ) from e

    return results


# ── Extractor factory ─────────────────────────────────────────────────────────
# This is the ONLY addition to this file.  All Ollama logic above is unchanged.

def get_extractor(method: str = "spacy"):
    """Return the appropriate extraction function based on method name.

    Args:
        method: "spacy" for fast NLP extraction (zero LLM calls)
                "llm"   for original Ollama-based extraction

    Returns:
        A callable with signature:
            fn(chunks: list, document_id: str) -> list[ExtractionResult]

    Raises:
        ValueError: If method is not recognised.
    """
    method = method.lower().strip()
    if method == "spacy":
        from .spacy_extractor import extract_entities_and_relationships_spacy
        return extract_entities_and_relationships_spacy
    elif method in ("llm", "ollama"):
        return extract_entities_and_relationships
    else:
        raise ValueError(
            f"Unknown extraction method '{method}'. "
            "Valid values: 'spacy', 'llm'."
        )
