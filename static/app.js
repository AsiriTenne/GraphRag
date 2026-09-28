document.addEventListener("DOMContentLoaded", () => {
  let selectedFile = null;
  let networkInstance = null;

  // ── Graph scope state ─────────────────────────────────────────────────────────
  let currentGraphDocId = null;   // null = show all, string = show one doc
  let currentGraphDocName = null; // display name for the scope label

  // ── DOM Elements ────────────────────────────────────────────────────────────
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("pdf-file-input");
  const selectedFileName = document.getElementById("selected-file-name");
  const processBtn = document.getElementById("process-btn");
  const extractionSelect = document.getElementById("extraction-method-select");
  const progressContainer = document.getElementById("progress-container");
  const progressBar = document.getElementById("progress-bar");
  const progressStatus = document.getElementById("progress-status");
  const progressPercent = document.getElementById("progress-percent");
  const alertBox = document.getElementById("alert-box");

  const neo4jBadge = document.getElementById("neo4j-badge");
  const openrouterBadge = document.getElementById("openrouter-badge");
  const extractionBadge = document.getElementById("extraction-badge");

  const statDocs = document.getElementById("stat-docs");
  const statChunks = document.getElementById("stat-chunks");
  const statEntities = document.getElementById("stat-entities");
  const statRels = document.getElementById("stat-rels");

  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const refreshGraphBtn = document.getElementById("refresh-graph-btn");
  const entitySearchInput = document.getElementById("entity-search-input");
  const entitySearchBtn = document.getElementById("entity-search-btn");
  const entitiesList = document.getElementById("entities-list");
  const docsList = document.getElementById("docs-list");

  // Chat elements
  const chatMessages = document.getElementById("chat-messages");
  const chatInput = document.getElementById("chat-input");
  const chatSendBtn = document.getElementById("chat-send-btn");
  const docFilterInput = document.getElementById("doc-filter-input");

  // ── 1. System Health ─────────────────────────────────────────────────────────
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();

      // Neo4j Status
      if (data.neo4j.status === "online") {
        neo4jBadge.className = "badge badge-online";
        neo4jBadge.innerHTML = `<span class="dot"></span> Neo4j: Online`;
      } else {
        neo4jBadge.className = "badge badge-offline";
        neo4jBadge.innerHTML = `<span class="dot"></span> Neo4j: Offline`;
      }

      // OpenRouter Status
      if (data.openrouter && data.openrouter.status === "configured") {
        openrouterBadge.className = "badge badge-online";
        openrouterBadge.innerHTML = `<span class="dot"></span> OpenRouter: ${data.openrouter.model.split("/").pop()}`;
      } else {
        openrouterBadge.className = "badge badge-offline";
        openrouterBadge.innerHTML = `<span class="dot"></span> OpenRouter: No Key`;
      }

      // Extraction method badge
      const method = data.extraction_method || "spacy";
      extractionBadge.className = method === "spacy" ? "badge badge-online" : "badge badge-info";
      extractionBadge.innerHTML = `<span class="icon">⚙️</span> Method: ${method}`;

      // Sync the dropdown with server default
      if (extractionSelect) extractionSelect.value = method;

    } catch (err) {
      neo4jBadge.className = "badge badge-offline";
      neo4jBadge.innerHTML = `<span class="dot"></span> Server Disconnected`;
    }
  }

  // ── 2. Stats ─────────────────────────────────────────────────────────────────
  async function loadStats() {
    try {
      const res = await fetch("/api/graph/stats");
      const data = await res.json();
      statDocs.textContent = data.document_nodes || 0;
      statChunks.textContent = data.chunk_nodes || 0;
      statEntities.textContent = data.entity_nodes || 0;
      statRels.textContent = data.total_relationships || 0;
    } catch (err) {
      console.error("Failed to load stats:", err);
    }
  }

  // ── 3. File Selection & Drag-and-Drop ────────────────────────────────────────
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFileSelected(e.target.files[0]);
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) handleFileSelected(e.dataTransfer.files[0]);
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      showAlert("Please select a valid PDF file.", "error");
      return;
    }
    selectedFile = file;
    selectedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    processBtn.disabled = false;
    hideAlert();
  }

  // ── 4. Upload & Pipeline ──────────────────────────────────────────────────────
  processBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    processBtn.disabled = true;
    progressContainer.classList.remove("hidden");
    hideAlert();
    updateProgress("Uploading PDF...", 10);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("extraction_method", extractionSelect.value);

    try {
      updateProgress("Running extraction pipeline...", 40);

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Failed to process PDF.");

      const method = data.extraction_method || "unknown";
      const totalTime = data.timings_seconds?.total?.toFixed(2) || "?";

      updateProgress("Complete!", 100);
      showAlert(
        `✅ Done! ` +
        `Neo4j: ${data.neo4j_chunks ?? data.chunk_count} chunks, ` +
        `${data.neo4j_entities ?? data.entity_count} entities, ` +
        `${data.neo4j_relationships ?? data.relationship_count} rels ` +
        `(extracted: ${data.entity_count}e / ${data.relationship_count}r, ` +
        `${data.vectors_stored || 0} vectors). ` +
        `Method: ${method} | Time: ${totalTime}s`,
        "success"
      );
      loadStats();
      loadDocuments();
      processBtn.disabled = false;

      // ── Auto-switch to Knowledge Graph tab and show this doc's graph ──────
      const newDocId = data.document_id;
      if (newDocId) {
        // Strip uuid prefix for a clean display name
        const rawName = data.filename || newDocId;
        const dispName = rawName.includes("_") ? rawName.split("_").slice(1).join("_") : rawName;
        currentGraphDocId = newDocId;
        currentGraphDocName = dispName;

        // Switch tab
        tabBtns.forEach((b) => b.classList.remove("active"));
        tabPanes.forEach((p) => p.classList.remove("active"));
        document.querySelector('[data-tab="graph-view"]').classList.add("active");
        document.getElementById("graph-view").classList.add("active");

        // Load the document-scoped graph
        loadGraphData(newDocId, dispName);
      }

    } catch (err) {
      updateProgress("Failed!", 0);
      showAlert(`Error: ${err.message}`, "error");
      processBtn.disabled = false;
    }
  });

  function updateProgress(message, percent) {
    progressStatus.textContent = message;
    progressPercent.textContent = `${percent}%`;
    progressBar.style.width = `${percent}%`;
  }

  function showAlert(msg, type) {
    alertBox.textContent = msg;
    alertBox.className = `alert-box alert-${type}`;
    alertBox.classList.remove("hidden");
  }

  function hideAlert() {
    alertBox.classList.add("hidden");
  }

  // ── 5. Chat ───────────────────────────────────────────────────────────────────
  function appendMessage(role, content, meta = null) {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-message chat-message--${role}`;

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.innerHTML = content;
    wrapper.appendChild(bubble);

    if (meta) {
      const metaEl = document.createElement("div");
      metaEl.className = "chat-meta";
      metaEl.innerHTML = meta;
      wrapper.appendChild(metaEl);
    }

    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendCitationPanel(chunks, graphFacts) {
    if (!chunks?.length && !graphFacts?.length) return;

    const panel = document.createElement("div");
    panel.className = "citation-panel";

    let html = `<details><summary>📎 Sources (${chunks.length} chunks, ${graphFacts.length} graph facts)</summary><div class="citation-list">`;

    for (const c of chunks) {
      html += `
        <div class="citation-item">
          <span class="citation-badge ${c.source === 'qdrant_semantic' ? 'badge-qdrant' : 'badge-neo4j'}">
            ${c.source === 'qdrant_semantic' ? '⚡ Qdrant' : '🔗 Neo4j'}
          </span>
          <strong>Page ${c.page_number}, Chunk ${c.chunk_index}</strong> — ${c.filename}<br>
          <span class="citation-text">${escapeHtml(c.text)}</span>
          <span class="citation-score">score: ${c.score}</span>
        </div>`;
    }

    for (const f of graphFacts) {
      html += `<div class="citation-item"><span class="citation-badge badge-graph">🕸 Graph</span>${escapeHtml(f)}</div>`;
    }

    html += `</div></details>`;
    panel.innerHTML = html;
    chatMessages.appendChild(panel);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function sendQuestion() {
    const question = chatInput.value.trim();
    if (!question) return;

    chatInput.value = "";
    chatSendBtn.disabled = true;

    appendMessage("user", escapeHtml(question));
    appendMessage("assistant", `<span class="typing-indicator">⏳ Retrieving evidence and generating answer…</span>`);

    const payload = { question };
    const docId = docFilterInput.value.trim();
    if (docId) payload.document_id = docId;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      // Remove typing indicator
      chatMessages.removeChild(chatMessages.lastChild);

      if (!res.ok) {
        appendMessage("assistant", `❌ ${escapeHtml(data.error || "Unknown error")}`);
        return;
      }

      const meta = [
        `⚡ Retrieval: ${data.retrieval_latency_s}s`,
        `🤖 Answer: ${data.answer_latency_s}s`,
        `📄 Chunks: ${data.chunks_retrieved}`,
        `🕸 Graph facts: ${data.graph_facts_retrieved}`,
        `🔢 LLM calls: ${data.llm_calls}`,
        `🪙 Tokens: ${data.prompt_tokens}↑ ${data.completion_tokens}↓`,
      ].join(" &nbsp;|&nbsp; ");

      appendMessage("assistant", escapeHtml(data.answer).replace(/\n/g, "<br>"), meta);
      appendCitationPanel(data.retrieved_chunks || [], data.graph_facts || []);

    } catch (err) {
      chatMessages.removeChild(chatMessages.lastChild);
      appendMessage("assistant", `❌ Network error: ${escapeHtml(err.message)}`);
    } finally {
      chatSendBtn.disabled = false;
      chatInput.focus();
    }
  }

  chatSendBtn.addEventListener("click", sendQuestion);
  chatInput.addEventListener("keyup", (e) => {
    if (e.key === "Enter") sendQuestion();
  });

  // ── 6. Graph Visualization ────────────────────────────────────────────────────
  const scopeLabel = document.getElementById("graph-scope-label");
  const showAllBtn = document.getElementById("show-all-graph-btn");

  // NODE COLORS by group
  const GROUP_COLORS = {
    Document:     { bg: "#f59e0b", border: "#fbbf24" }, // amber — Document hub
    Person:       { bg: "#6366f1", border: "#818cf8" },
    Organization: { bg: "#10b981", border: "#34d399" },
    Location:     { bg: "#ef4444", border: "#f87171" },
    GPE:          { bg: "#ef4444", border: "#f87171" },
    Date:         { bg: "#8b5cf6", border: "#a78bfa" },
    Entity:       { bg: "#6366f1", border: "#818cf8" },
    _default:     { bg: "#6366f1", border: "#818cf8" },
  };

  function nodeColor(group) {
    const c = GROUP_COLORS[group] || GROUP_COLORS._default;
    return { background: c.bg, border: c.border, highlight: { background: "#06b6d4", border: "#67e8f9" } };
  }

  async function loadGraphData(docId, docName) {
    // Update module-level state so Refresh + tab-click stay in sync
    if (docId !== undefined) {
      currentGraphDocId = docId || null;
      currentGraphDocName = docName || null;
    }

    const container = document.getElementById("network-canvas");
    container.innerHTML = `<div class="canvas-placeholder">Loading graph…</div>`;

    // Update scope label + Show All button
    if (currentGraphDocId) {
      const name = currentGraphDocName || currentGraphDocId;
      scopeLabel.textContent = `📄 Viewing: ${name}`;
      showAllBtn.style.display = "inline-flex";
    } else {
      scopeLabel.textContent = `🌍 All documents`;
      showAllBtn.style.display = "none";
    }

    const url = currentGraphDocId
      ? `/api/graph/data?limit=150&document_id=${encodeURIComponent(currentGraphDocId)}`
      : `/api/graph/data?limit=150`;

    try {
      const res = await fetch(url);
      const data = await res.json();

      if (!data.nodes || data.nodes.length === 0) {
        container.innerHTML = `<div class="canvas-placeholder">No graph data found for this document.</div>`;
        return;
      }

      const visNodes = data.nodes.map((n) => {
        const isDoc = n.group === "Document";
        return {
          id: n.id,
          label: n.label.length > 24 ? n.label.slice(0, 22) + "…" : n.label,
          title: n.title || n.label,   // tooltip on hover
          shape: isDoc ? "star" : "dot",
          size: isDoc ? 28 : 14,
          color: nodeColor(n.group),
          font: { color: "#f8fafc", face: "Inter", size: isDoc ? 13 : 11 },
          borderWidth: isDoc ? 3 : 1,
        };
      });

      const visEdges = data.edges.map((e) => ({
        from: e.from,
        to: e.to,
        label: e.label,
        title: e.title || "",
        arrows: "to",
        color: { color: "#06b6d4", opacity: 0.7, highlight: "#38bdf8" },
        font: { color: "#94a3b8", size: 9, align: "middle" },
        smooth: { type: "dynamic" },
      }));

      const networkData = {
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges),
      };

      const options = {
        physics: {
          stabilization: { iterations: 150 },
          barnesHut: { gravitationalConstant: -4000, springLength: 140, damping: 0.15 },
        },
        interaction: { hover: true, tooltipDelay: 150, navigationButtons: true },
        nodes: { scaling: { min: 10, max: 30 } },
      };

      container.innerHTML = "";
      networkInstance = new vis.Network(container, networkData, options);

      // Show node details on click (in tooltip-style)
      networkInstance.on("click", (params) => {
        if (params.nodes.length > 0) {
          const nodeId = params.nodes[0];
          const node = data.nodes.find((n) => n.id === nodeId);
          if (node) {
            const tip = node.title || node.label;
            scopeLabel.textContent = `Selected: ${tip}`;
          }
        } else if (currentGraphDocId) {
          scopeLabel.textContent = `📄 Viewing: ${currentGraphDocName || currentGraphDocId}`;
        } else {
          scopeLabel.textContent = `🌍 All documents`;
        }
      });

    } catch (err) {
      container.innerHTML = `<div class="canvas-placeholder">Failed to render graph: ${err.message}</div>`;
    }
  }

  refreshGraphBtn.addEventListener("click", () => loadGraphData());

  showAllBtn.addEventListener("click", () => {
    currentGraphDocId = null;
    currentGraphDocName = null;
    loadGraphData();
  });

  // ── 7. Entity Search ──────────────────────────────────────────────────────────
  async function performEntitySearch() {
    const query = entitySearchInput.value.trim();
    try {
      const res = await fetch(`/api/entities?q=${encodeURIComponent(query)}`);
      const results = await res.json();

      if (results.length === 0) {
        entitiesList.innerHTML = `<div class="empty-state">No matching entities found.</div>`;
        return;
      }

      entitiesList.innerHTML = results.map((e) => `
        <div class="entity-item">
          <div>
            <div>
              <span class="entity-name">${e.name}</span>
              <span class="entity-type">${e.entity_type || "Entity"}</span>
            </div>
            <div class="entity-desc">${e.description || "No description available."}</div>
          </div>
          <div>
            <span class="stat-tag">${(e.connections || []).length} connection(s)</span>
          </div>
        </div>
      `).join("");
    } catch (err) {
      entitiesList.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
    }
  }

  entitySearchBtn.addEventListener("click", performEntitySearch);
  entitySearchInput.addEventListener("keyup", (e) => {
    if (e.key === "Enter") performEntitySearch();
  });

  // ── 8. Documents List ─────────────────────────────────────────────────────────
  async function loadDocuments() {
    try {
      const res = await fetch("/api/documents");
      const docs = await res.json();

      if (docs.length === 0) {
        docsList.innerHTML = `<div class="empty-state">No documents found in Neo4j.</div>`;
        return;
      }

      docsList.innerHTML = docs.map((d) => `
        <div class="doc-item">
          <div>
            <div style="font-weight:600; color:#f8fafc;">${d.filename}</div>
            <div style="font-size:12px; color:#94a3b8;">
              ID: ${d.document_id} &nbsp;|&nbsp; Pages: ${d.total_pages} &nbsp;|&nbsp;
              Method: ${d.extraction_method || "unknown"}
            </div>
            <div style="font-size:12px; color:#64748b; margin-top:3px;">
              Neo4j: ${d.entity_count ?? "?"} entities &nbsp;·&nbsp;
              ${d.relationship_count ?? "?"} rels
            </div>
          </div>
          <div>
            <span class="card-tag">${d.chunk_count} Chunks</span>
          </div>
        </div>
      `).join("");
    } catch (err) {
      docsList.innerHTML = `<div class="empty-state">Failed to load documents: ${err.message}</div>`;
    }
  }

  // ── 9. Tab Switching ──────────────────────────────────────────────────────────
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetTab = btn.getAttribute("data-tab");
      document.getElementById(targetTab).classList.add("active");

      // Graph tab: respect current document scope
      if (targetTab === "graph-view") loadGraphData();
      if (targetTab === "entities-view") performEntitySearch();
      if (targetTab === "docs-view") loadDocuments();
    });
  });

  // ── Initialize ────────────────────────────────────────────────────────────────
  checkHealth();
  loadStats();
  loadDocuments();
});
