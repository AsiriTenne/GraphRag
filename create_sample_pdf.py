"""Generate a sample PDF file for end-to-end GraphRAG pipeline demonstration."""

from pathlib import Path
from fpdf import FPDF
from config.settings import UPLOAD_DIR

def generate_sample_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Knowledge Graph Construction and GraphRAG Architecture", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 11)
    page_1_text = (
        "Retrieval-Augmented Generation (RAG) enhances Large Language Models by anchoring responses in "
        "external knowledge bases. Traditional RAG relies heavily on vector similarity search over text chunks. "
        "However, GraphRAG integrates Knowledge Graphs constructed using graph databases like Neo4j to preserve "
        "complex semantic relationships between entities.\n\n"
        "Neo4j is an enterprise-grade graph database developed by Neo4j Inc. It utilizes Cypher, a declarative "
        "graph query language, to inspect nodes, edges, and multi-hop paths. In GraphRAG architectures, entities "
        "such as Organizations, Technologies, Concepts, and Persons are represented as distinct nodes linked by "
        "typed relationships like USES, CREATED, or PART_OF."
    )
    pdf.multi_cell(0, 7, page_1_text)
    
    # Page 2
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "LLM Extraction and Neo4j Persistence", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 11)
    page_2_text = (
        "OpenAI produces state-of-the-art language models including GPT-4o and GPT-4o-mini. The extraction pipeline "
        "sends parsed PDF text chunks to OpenAI API endpoints to extract canonical entities and directed relationships. "
        "The entity resolver deduplicates entities across chunks and attributes provenance.\n\n"
        "Antigravity is an advanced agentic coding assistant developed by Google DeepMind. It automates full-stack "
        "development tasks and pair programming workflows. The complete pipeline persists Document nodes, Chunk nodes, "
        "Entity nodes, and MENTIONS edges into Neo4j for hybrid retrieval."
    )
    pdf.multi_cell(0, 7, page_2_text)
    
    output_path = UPLOAD_DIR / "AI_Knowledge_Graph_Research.pdf"
    pdf.output(str(output_path))
    print(f"Sample PDF created at: {output_path}")

if __name__ == "__main__":
    generate_sample_pdf()
