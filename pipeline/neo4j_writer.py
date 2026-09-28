"""Neo4j database writer & query module.

Handles all Neo4j database operations: creating document nodes,
chunk nodes, entity nodes, relationships, and safe query execution.
"""

from __future__ import annotations

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable


from config.settings import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from .pdf_loader import PDFContent
from .chunker import Chunk
from .entity_resolver import ResolvedEntity


def get_driver():
    """Create and return a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def check_neo4j_connection() -> tuple[bool, str]:
    """Check if the Neo4j database is reachable and credentials are valid.

    Returns:
        (is_connected, message)
    """
    try:
        driver = get_driver()
        driver.verify_connectivity()
        with driver.session() as session:
            session.run("RETURN 1 AS test")
        driver.close()
        return True, "Connected to Neo4j successfully."
    except Exception as e:
        return False, f"Failed to connect to Neo4j at {NEO4J_URI}: {str(e)}"


def write_to_neo4j(
    content: PDFContent,
    chunks: list[Chunk],
    resolved_entities: list[ResolvedEntity],
    resolved_relationships: list[dict],
    document_id: str,
    chunk_mentions: list[dict] = None,
    extraction_method: str = "spacy",
) -> dict:
    """Write all extracted data to Neo4j.

    Creates:
        - Document node
        - Chunk nodes linked to Document via HAS_CHUNK
        - Entity nodes
        - MENTIONS relationships (Chunk -> Entity)
        - Extracted relationships (Entity -> Entity)

    Returns:
        Dict with counts of created nodes and relationships.

    Raises:
        ConnectionError: If Neo4j is offline or unreachable.
    """
    is_connected, conn_msg = check_neo4j_connection()
    if not is_connected:
        raise ConnectionError(
            f"Cannot write knowledge graph. Neo4j is offline or unreachable: {conn_msg}"
        )

    driver = get_driver()
    stats = {
        "document_nodes": 0,
        "chunk_nodes": 0,
        "entity_nodes": 0,
        "mentions_relationships": 0,
        "entity_relationships": 0,
    }

    try:
        with driver.session() as session:
            # 1. Create Document node
            session.run(
                """
                MERGE (d:Document {document_id: $document_id})
                SET d.filename = $filename,
                    d.total_pages = $total_pages,
                    d.extraction_method = $extraction_method,
                    d.created_at = datetime()
                """,
                document_id=document_id,
                filename=content.filename,
                total_pages=content.total_pages,
                extraction_method=extraction_method,
            )
            stats["document_nodes"] = 1

            # 2. Create Chunk nodes and link to Document
            for chunk in chunks:
                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.text = $text,
                        c.page_number = $page_number,
                        c.chunk_index = $chunk_index,
                        c.document_id = $document_id
                    WITH c
                    MATCH (d:Document {document_id: $document_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    document_id=document_id,
                )
                stats["chunk_nodes"] += 1

            # 3. Create Entity nodes
            for entity in resolved_entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.entity_type = $entity_type,
                        e.description = $description,
                        e.extraction_method = $extraction_method
                    """,
                    name=entity.canonical_name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    extraction_method=extraction_method,
                )
                stats["entity_nodes"] += 1

            # 4. Create MENTIONS relationships (Chunk -> Entity)
            mentions_created: set[tuple[str, str]] = set()

            # Process explicit chunk_mentions passed from resolver
            if chunk_mentions:
                for mention in chunk_mentions:
                    chunk_id = mention.get("chunk_id")
                    entity_name = mention.get("entity_name")
                    if chunk_id and entity_name:
                        key = (chunk_id, entity_name)
                        if key not in mentions_created:
                            session.run(
                                """
                                MATCH (c:Chunk {chunk_id: $chunk_id})
                                MATCH (e:Entity {name: $entity_name})
                                MERGE (c)-[:MENTIONS]->(e)
                                """,
                                chunk_id=chunk_id,
                                entity_name=entity_name,
                            )
                            mentions_created.add(key)
                            stats["mentions_relationships"] += 1

            # Also ensure relationships sources/targets are linked to chunk
            for rel in resolved_relationships:
                chunk_id = rel.get("chunk_id")
                if chunk_id:
                    for name in [rel.get("source"), rel.get("target")]:
                        if name:
                            key = (chunk_id, name)
                            if key not in mentions_created:
                                session.run(
                                    """
                                    MATCH (c:Chunk {chunk_id: $chunk_id})
                                    MATCH (e:Entity {name: $entity_name})
                                    MERGE (c)-[:MENTIONS]->(e)
                                    """,
                                    chunk_id=chunk_id,
                                    entity_name=name,
                                )
                                mentions_created.add(key)
                                stats["mentions_relationships"] += 1

            # 5. Create Entity -> Entity relationships
            for rel in resolved_relationships:
                rel_type = rel["relationship"].replace(" ", "_").upper()
                rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_")
                if not rel_type:
                    rel_type = "ASSOCIATED_WITH"

                session.run(
                    f"""
                    MATCH (source:Entity {{name: $source_name}})
                    MATCH (target:Entity {{name: $target_name}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r.description = $description,
                        r.document_id = $document_id,
                        r.extraction_method = $extraction_method
                    """,
                    source_name=rel["source"],
                    target_name=rel["target"],
                    description=rel.get("description", ""),
                    document_id=document_id,
                    extraction_method=extraction_method,
                )
                stats["entity_relationships"] += 1

    finally:
        driver.close()

    return stats


def get_graph_stats() -> dict:
    """Fetch counts of documents, chunks, entities, and relationships in Neo4j."""
    is_connected, _ = check_neo4j_connection()
    if not is_connected:
        return {
            "status": "offline",
            "document_nodes": 0,
            "chunk_nodes": 0,
            "entity_nodes": 0,
            "total_relationships": 0,
        }

    driver = get_driver()
    try:
        with driver.session() as session:
            doc_count = session.run("MATCH (d:Document) RETURN count(d) AS count").single()["count"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

            return {
                "status": "online",
                "document_nodes": doc_count,
                "chunk_nodes": chunk_count,
                "entity_nodes": entity_count,
                "total_relationships": rel_count,
            }
    finally:
        driver.close()


def get_documents() -> list[dict]:
    """Fetch list of all processed documents with per-document stats."""
    is_connected, _ = check_neo4j_connection()
    if not is_connected:
        return []

    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) AS chunk_count
                RETURN d.document_id AS document_id,
                       d.filename AS filename,
                       d.total_pages AS total_pages,
                       d.created_at AS created_at,
                       d.extraction_method AS extraction_method,
                       chunk_count
                ORDER BY d.created_at DESC
                """
            )
            docs_raw = [dict(record) for record in result]

        # Serialize Neo4j DateTime -> ISO string (Flask's jsonify can't handle it)
        docs = []
        for doc in docs_raw:
            created = doc.get("created_at")
            if created is not None and hasattr(created, "isoformat"):
                doc["created_at"] = created.isoformat()
            elif created is not None:
                doc["created_at"] = str(created)
            docs.append(doc)

        # Enrich each document with per-document entity and relationship counts
        with driver.session() as session:
            for doc in docs:
                doc_id = doc["document_id"]

                # Distinct entities reachable from this document's chunks
                ent_count = session.run(
                    """
                    MATCH (c:Chunk {document_id: $doc_id})-[:MENTIONS]->(e:Entity)
                    RETURN count(DISTINCT e) AS cnt
                    """,
                    doc_id=doc_id,
                ).single()["cnt"]

                # Entity->Entity rels tagged with this document_id
                rel_count = session.run(
                    """
                    MATCH (e1:Entity)-[r]->(e2:Entity)
                    WHERE r.document_id = $doc_id
                    RETURN count(r) AS cnt
                    """,
                    doc_id=doc_id,
                ).single()["cnt"]

                doc["entity_count"] = ent_count
                doc["relationship_count"] = rel_count

        return docs

    finally:
        driver.close()


def get_document_stats(document_id: str) -> dict:
    """Return per-document Neo4j write stats for verification after ingestion.

    Returns a dict with chunk_count, entity_count, mention_count,
    and relationship_count reflecting what is actually stored in Neo4j
    for this specific document_id.
    """
    is_connected, _ = check_neo4j_connection()
    if not is_connected:
        return {}

    driver = get_driver()
    try:
        with driver.session() as session:
            chunk_count = session.run(
                "MATCH (c:Chunk {document_id: $doc_id}) RETURN count(c) AS cnt",
                doc_id=document_id,
            ).single()["cnt"]

            mention_count = session.run(
                """
                MATCH (c:Chunk {document_id: $doc_id})-[:MENTIONS]->(e:Entity)
                RETURN count(*) AS cnt
                """,
                doc_id=document_id,
            ).single()["cnt"]

            entity_count = session.run(
                """
                MATCH (c:Chunk {document_id: $doc_id})-[:MENTIONS]->(e:Entity)
                RETURN count(DISTINCT e) AS cnt
                """,
                doc_id=document_id,
            ).single()["cnt"]

            rel_count = session.run(
                """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE r.document_id = $doc_id
                RETURN count(r) AS cnt
                """,
                doc_id=document_id,
            ).single()["cnt"]

            return {
                "document_id": document_id,
                "chunks_in_neo4j": chunk_count,
                "mentions_in_neo4j": mention_count,
                "entities_reachable": entity_count,
                "relationships_in_neo4j": rel_count,
            }
    finally:
        driver.close()


def get_graph_data(limit: int = 150, document_id: str = None) -> dict:
    """Retrieve nodes and relationships for UI graph visualization.

    Args:
        limit: Maximum number of Entity→Entity edges to return.
        document_id: If provided, restrict the graph to this document only.
                     Returns the Document node, all Entity nodes reachable from
                     its chunks, and all Entity→Entity rels tagged to it.
                     If None, returns all Entity→Entity rels across all documents.
    """
    is_connected, _ = check_neo4j_connection()
    if not is_connected:
        return {"nodes": [], "edges": [], "document_id": document_id}

    driver = get_driver()
    try:
        nodes_dict: dict[str, dict] = {}
        edges_list: list[dict] = []

        with driver.session() as session:

            if document_id:
                # ── Document-scoped graph ──────────────────────────────────────
                # 1. Fetch Document node metadata
                doc_row = session.run(
                    """
                    MATCH (d:Document {document_id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                    RETURN d.filename AS filename, count(c) AS chunk_count
                    """,
                    doc_id=document_id,
                ).single()

                if doc_row:
                    fname = (doc_row["filename"] or document_id)
                    # Strip uuid prefix from filename if present (e.g. "abc12345_Report.pdf" → "Report.pdf")
                    display_name = fname.split("_", 1)[-1] if "_" in fname else fname
                    doc_node_id = f"__doc__{document_id}"
                    nodes_dict[doc_node_id] = {
                        "id": doc_node_id,
                        "label": display_name,
                        "group": "Document",
                        "shape": "star",
                        "title": f"Document: {fname}\n{doc_row['chunk_count']} chunks",
                    }

                # 2. All Entity nodes reachable via MENTIONS from this doc's chunks
                ent_result = session.run(
                    """
                    MATCH (c:Chunk {document_id: $doc_id})-[:MENTIONS]->(e:Entity)
                    RETURN DISTINCT e.name AS name, e.entity_type AS etype
                    ORDER BY e.name
                    LIMIT 200
                    """,
                    doc_id=document_id,
                )
                for rec in ent_result:
                    ename = rec["name"]
                    if ename not in nodes_dict:
                        nodes_dict[ename] = {
                            "id": ename,
                            "label": ename,
                            "group": rec["etype"] or "Entity",
                            "shape": "dot",
                        }

                # 3. Entity→Entity rels tagged with this document_id
                rel_result = session.run(
                    """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE r.document_id = $doc_id
                    RETURN s.name AS source,
                           s.entity_type AS source_type,
                           t.name AS target,
                           t.entity_type AS target_type,
                           type(r) AS rel_type,
                           r.description AS description
                    LIMIT $limit
                    """,
                    doc_id=document_id,
                    limit=limit,
                )
                for record in rel_result:
                    s_name = record["source"]
                    t_name = record["target"]
                    # Ensure both endpoints are in nodes_dict (should already be)
                    if s_name not in nodes_dict:
                        nodes_dict[s_name] = {"id": s_name, "label": s_name, "group": record["source_type"] or "Entity", "shape": "dot"}
                    if t_name not in nodes_dict:
                        nodes_dict[t_name] = {"id": t_name, "label": t_name, "group": record["target_type"] or "Entity", "shape": "dot"}
                    edges_list.append({
                        "from": s_name,
                        "to": t_name,
                        "label": record["rel_type"].replace("_", " ").title(),
                        "title": record["description"] or "",
                    })

            else:
                # ── Global graph (all documents) ───────────────────────────────
                result = session.run(
                    """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    RETURN s.name AS source,
                           s.entity_type AS source_type,
                           t.name AS target,
                           t.entity_type AS target_type,
                           type(r) AS rel_type,
                           r.description AS description
                    LIMIT $limit
                    """,
                    limit=limit,
                )
                for record in result:
                    s_name = record["source"]
                    t_name = record["target"]
                    if s_name not in nodes_dict:
                        nodes_dict[s_name] = {"id": s_name, "label": s_name, "group": record["source_type"] or "Entity", "shape": "dot"}
                    if t_name not in nodes_dict:
                        nodes_dict[t_name] = {"id": t_name, "label": t_name, "group": record["target_type"] or "Entity", "shape": "dot"}
                    edges_list.append({
                        "from": s_name,
                        "to": t_name,
                        "label": record["rel_type"].replace("_", " ").title(),
                        "title": record["description"] or "",
                    })

        return {
            "nodes": list(nodes_dict.values()),
            "edges": edges_list,
            "document_id": document_id,
        }
    finally:
        driver.close()



def search_entities(query: str) -> list[dict]:
    """Search for entities matching a keyword and return their details & relationships."""
    is_connected, _ = check_neo4j_connection()
    if not is_connected or not query.strip():
        return []

    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                   OR toLower(e.entity_type) CONTAINS toLower($query)
                OPTIONAL MATCH (e)-[r]-(other:Entity)
                RETURN e.name AS name,
                       e.entity_type AS entity_type,
                       e.description AS description,
                       collect(DISTINCT {
                           type: type(r),
                           target: other.name
                       }) AS connections
                LIMIT 20
                """,
                query=query.strip(),
            )
            return [dict(record) for record in result]
    finally:
        driver.close()


def clear_database() -> dict:
    """Wipe all nodes and relationships from Neo4j database."""
    is_connected, msg = check_neo4j_connection()
    if not is_connected:
        raise ConnectionError(f"Cannot clear database. Neo4j is offline: {msg}")

    driver = get_driver()
    try:
        with driver.session() as session:
            res_nodes = session.run("MATCH (n) DETACH DELETE n")
            summary = res_nodes.consume()
            return {"status": "cleared", "nodes_deleted": summary.counters.nodes_deleted}
    finally:
        driver.close()

