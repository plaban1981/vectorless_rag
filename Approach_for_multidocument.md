# Multi-Document PageIndex RAG — Architecture & Approaches

> **Context:** The current Vectorless RAG implementation uses PageIndex to navigate a single PDF document via a hierarchical tree index built by Gemini. This document analyses why the current design breaks for multiple documents and proposes three concrete architectural approaches to extend it, ranked by complexity.

---

## Table of Contents

1. [Why the Current Implementation Breaks](#1-why-the-current-implementation-breaks)
2. [What Needs to Change](#2-what-needs-to-change)
3. [Approach 1 — Concatenated Trees](#3-approach-1--concatenated-trees-simple)
4. [Approach 2 — Two-Level Meta-Tree Navigation](#4-approach-2--two-level-meta-tree-navigation-recommended)
5. [Approach 3 — Parallel Agentic Chains](#5-approach-3--parallel-agentic-chains-powerful)
6. [The Synthesis Problem](#6-the-synthesis-problem)
7. [Comparison Matrix](#7-comparison-matrix)
8. [Recommended Migration Path](#8-recommended-migration-path)

---

## 1. Why the Current Implementation Breaks

### 1.1 `DocumentManager` is a Single-Document Singleton

The core state container holds exactly one document at a time. Uploading a second PDF silently overwrites the first.

```python
# vectorless_rag/document_manager.py

class DocumentManager:
    def __init__(self):
        self.document_name: str = ""
        self.pages: dict[int, str] = {}       # ← one document's pages
        self.tree_index: Optional[dict] = None # ← one document's tree
        self.tree_text: str = ""               # ← one document's formatted index
        self._page_image_paths: dict[int, Path] = {}

# Global singleton — shared by all chains
doc_manager = DocumentManager()
```

`load_pdf()` replaces every field in place. There is no concept of a document collection, a registry, or a persistent multi-doc store.

### 1.2 Both RAG Chains Only See One `tree_text`

```python
# vectorless_rag/chains/agentic_rag.py

messages = [
    HumanMessage(content=(
        f"Here is the document tree index:\n\n{self._doc.tree_text}\n\n"
        # ↑ single document's tree only
        f"QUESTION: {query}"
    )),
]
```

If the answer spans two PDFs, the LLM never sees the second document's tree. It can only navigate and `FETCH_NODE` from one tree, making cross-document answers structurally impossible.

### 1.3 Node IDs Collide Across Documents

Every document Gemini indexes produces nodes named `N001`, `N002`, `N003`, and so on. With two documents loaded simultaneously, `find_node("N001")` is ambiguous — it could belong to either document.

```python
# document_manager.py
def _find_node_recursive(self, node_id: str, nodes: list) -> Optional[dict]:
    for node in nodes:
        if node["node_id"] == node_id:  # ← which document's N001?
            return node
        ...
```

### 1.4 Page Numbers Are Document-Relative

`get_pages(start_page, end_page)` indexes into `self.pages` which is keyed by page number within a single PDF. With multiple documents, page 12 in Document A and page 12 in Document B are different things.

### 1.5 The `/api/upload` Endpoint Replaces State

```python
# app.py
result = await loop.run_in_executor(
    executor, pipeline.load_document, file_bytes, file.filename
)
# Every new upload destroys the previous document's state
```

There is no append-to-collection behaviour — only replace.

---

## 2. What Needs to Change

Before choosing an approach, these components must be modified regardless:

| Component | Current Behaviour | Required Change |
|-----------|------------------|-----------------|
| `DocumentManager` | Holds 1 document | Hold a `dict[str, DocumentManager]` keyed by doc ID |
| Node IDs | `N001`, `N002` | Namespaced: `DOC1:N001`, `DOC2:N001` |
| `find_node(node_id)` | Searches 1 tree | Parse `DOC_ID:NODE_ID`, route to correct `DocumentManager` |
| `get_pages(start, end)` | 1 doc's PyMuPDF text | Accept `doc_id` parameter |
| `AgenticRAGChain` | 1 tree in context | Multi-doc tree context or meta-tree |
| `VisionRAGChain` | 1 doc's page images | Route image fetches by `doc_id` |
| `POST /api/upload` | Replaces current doc | Appends to collection |
| `GET /api/status` | Single doc info | List of all loaded documents |
| `GET /api/tree` | Single doc tree | Accept `doc_id` param or return all trees |

---

## 3. Approach 1 — Concatenated Trees (Simple)

### Core Idea

Keep the existing agentic loop unchanged. Merge all documents' `tree_text` into one combined context string, prefixing every node ID with a document identifier.

### How the Combined Tree Looks

```
═══════════════════════════════════════════════════════
DOCUMENT 1: Annual Report 2024  [doc_id: annual_2024]
Total pages: 60
═══════════════════════════════════════════════════════

[annual_2024:N001] Executive Summary (pages 1-3)
  Summary: Record revenue $4.2B ↑18% YoY. Gross margin 62%.
           Three acquisitions completed. See annual_2024:N004
           for full financial breakdown.

[annual_2024:N004] Financial Performance (pages 12-28)
  Summary: Q3 revenue $1.1B, APAC growth 34%...

  [annual_2024:N004a] Revenue by Region (pages 14-17)
    Summary: APAC $420M, EMEA $310M, Americas $370M...

═══════════════════════════════════════════════════════
DOCUMENT 2: Competitor Analysis 2024  [doc_id: comp_2024]
Total pages: 42
═══════════════════════════════════════════════════════

[comp_2024:N001] Market Overview (pages 1-5)
  Summary: Industry TAM $180B. Top 3 players hold 67% share...

[comp_2024:N003] Competitor Benchmarks (pages 14-20)
  Summary: Competitor A gross margin 58%, revenue $2.1B...
```

### How the Agentic Loop Uses It

The LLM sees the combined tree and navigates across documents in the same loop:

```json
{
  "action": "FETCH_NODE",
  "node_id": "annual_2024:N004a",
  "reasoning": "Need our revenue by region before comparing to competitor data in comp_2024:N003"
}
```

Then:

```json
{
  "action": "FETCH_NODE",
  "node_id": "comp_2024:N003",
  "reasoning": "Now fetching competitor benchmarks to compare against our APAC $420M figure"
}
```

Then:

```json
{
  "action": "ANSWER",
  "answer": "Our APAC revenue ($420M) leads Competitor A ($380M) by 10.5%...",
  "sources": ["annual_2024:N004a (pages 14-17)", "comp_2024:N003 (pages 14-20)"]
}
```

### Code Changes Required

```python
# New: DocumentCollection replaces the singleton DocumentManager

class DocumentCollection:
    def __init__(self):
        self._docs: dict[str, DocumentManager] = {}

    def add_document(self, file_bytes: bytes, filename: str) -> dict:
        doc_id = hashlib.sha256(file_bytes).hexdigest()[:12]
        dm = DocumentManager()
        dm.load_pdf(file_bytes, filename)
        self._docs[doc_id] = dm
        return {"doc_id": doc_id, **dm.metadata()}

    def combined_tree_text(self) -> str:
        parts = []
        for doc_id, dm in self._docs.items():
            # Prefix all node IDs with doc_id
            namespaced = _namespace_tree(dm.tree_text, doc_id)
            parts.append(
                f"{'=' * 55}\n"
                f"DOCUMENT: {dm.document_name}  [doc_id: {doc_id}]\n"
                f"{'=' * 55}\n\n"
                + namespaced
            )
        return "\n\n".join(parts)

    def find_node(self, namespaced_id: str) -> Optional[dict]:
        # "annual_2024:N004a" → doc_id="annual_2024", node_id="N004a"
        if ":" not in namespaced_id:
            return None
        doc_id, node_id = namespaced_id.split(":", 1)
        dm = self._docs.get(doc_id)
        return dm.find_node(node_id) if dm else None

    def get_pages(self, namespaced_id: str, start: int, end: int) -> str:
        doc_id, _ = namespaced_id.split(":", 1)
        dm = self._docs.get(doc_id)
        return dm.get_pages(start, end) if dm else ""
```

### Pros and Cons

| ✅ Pros | ❌ Cons |
|--------|--------|
| Minimal change to agentic loop logic | Context window grows with every document |
| Cross-document reasoning in a single LLM pass | 10 docs × 3,000-char trees = ~30k tokens just for the index |
| Full source attribution across docs | Gemini may lose focus on the right document in large combined trees |
| Existing FETCH_NODE / ANSWER format unchanged | Node ID parsing adds fragility |

### Best For

- **2–5 documents** with concise tree indexes
- Quick prototyping of multi-doc capability
- Use cases where documents are closely related (same domain, same time period)

---

## 4. Approach 2 — Two-Level Meta-Tree Navigation (Recommended)

### Core Idea

Build a **meta-index** — a tree of documents — and navigate in two distinct hops:

1. **Hop 1:** LLM reads the meta-index and selects which document(s) contain the answer
2. **Hop 2:** LLM navigates the selected document's PageIndex tree using the existing FETCH_NODE loop

This mirrors how a human navigates a library: first pick the right book off the shelf, then find the right chapter inside it.

### The Meta-Index Structure

```json
{
  "corpus": {
    "description": "Financial document collection — FY2024",
    "total_documents": 4,
    "documents": [
      {
        "doc_id": "annual_2024",
        "filename": "Annual Report 2024.pdf",
        "total_pages": 60,
        "abstract": "Full-year financial results. Revenue $4.2B ↑18% YoY.
                     Sections cover P&L, regional breakdown, risk factors,
                     board decisions. Key figures: gross margin 62%,
                     APAC revenue $420M, three acquisitions completed.",
        "key_topics": ["revenue", "APAC", "acquisitions", "risk", "Q3", "margin"]
      },
      {
        "doc_id": "comp_2024",
        "filename": "Competitor Analysis 2024.pdf",
        "total_pages": 42,
        "abstract": "External analysis of top 3 competitors. Covers market
                     share, pricing benchmarks, product gaps, and SWOT.
                     Industry TAM $180B. Competitor A margin 58%.",
        "key_topics": ["competitor", "market share", "benchmark", "pricing", "SWOT"]
      }
    ]
  }
}
```

The abstract for each document is auto-generated by Gemini during ingestion — a condensed, keyword-rich summary of the whole document (not a chapter, the whole thing).

### Two-Level Navigation Flow

```
Query: "How does our APAC revenue compare to Competitor A?"
         │
         ▼
  ┌─────────────────────┐
  │   META-INDEX READ   │   LLM reads corpus-level index
  │                     │   → selects: ["annual_2024", "comp_2024"]
  └─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ DOC 1  │ │  DOC 2   │   Parallel or sequential PageIndex navigation
│ Tree   │ │  Tree    │   for each selected document
│ Nav    │ │  Nav     │
└────────┘ └──────────┘
    │             │
    │  N004a      │  N003
    │  pages 14-17│  pages 14-20
    ▼             ▼
┌──────────────────────────┐
│      SYNTHESIZE          │   Final Gemini call merges both answers
│  Our APAC $420M vs       │
│  Competitor A $380M      │
└──────────────────────────┘
```

### Code Structure

```python
class MetaTreeNavigator:
    """
    Level-1 navigator: reads the corpus meta-index and selects
    relevant documents. Returns a list of doc_ids.
    """

    _META_SYSTEM = """\
    You are a document routing agent. Given a corpus of documents
    and a question, identify which document(s) most likely contain
    the answer.

    Return ONLY valid JSON:
    {
      "reasoning": "brief explanation",
      "doc_ids": ["doc_id_1", "doc_id_2"]
    }

    Select 1-3 documents maximum. Prefer specificity over coverage.
    """

    def select_documents(self, query: str, corpus_index: dict) -> list[str]:
        meta_text = self._format_corpus_index(corpus_index)
        response = self._llm.invoke([
            SystemMessage(content=self._META_SYSTEM),
            HumanMessage(content=(
                f"Question: {query}\n\n"
                f"Corpus:\n{meta_text}\n\n"
                "Return JSON with doc_ids to search."
            )),
        ])
        result = _parse_json(_content_to_str(response.content))
        return result.get("doc_ids", [])

    def _format_corpus_index(self, corpus: dict) -> str:
        lines = [f"CORPUS: {corpus['description']}\n"]
        for doc in corpus["documents"]:
            lines.append(f"[{doc['doc_id']}] {doc['filename']}")
            lines.append(f"  Pages: {doc['total_pages']}")
            lines.append(f"  Abstract: {doc['abstract']}")
            lines.append(f"  Topics: {', '.join(doc['key_topics'])}")
            lines.append("")
        return "\n".join(lines)


class MultiDocAgenticChain(Runnable):
    """
    Full two-level PageIndex chain for multiple documents.
    """

    def __init__(self, collection: DocumentCollection):
        self._collection = collection
        self._meta_nav   = MetaTreeNavigator()
        self._llm        = ChatGoogleGenerativeAI(...)

    def invoke(self, input: dict, config=None) -> dict:
        query = input["query"]

        # Level 1: select documents
        doc_ids = self._meta_nav.select_documents(
            query, self._collection.corpus_index
        )

        # Level 2: navigate each selected document's tree
        per_doc_results = {}
        for doc_id in doc_ids:
            dm     = self._collection.get(doc_id)
            chain  = AgenticRAGChain(dm)
            result = chain.invoke({"query": query})
            per_doc_results[doc_id] = result

        # Level 3: synthesize
        return self._synthesize(query, per_doc_results)

    def _synthesize(self, query: str, results: dict) -> dict:
        context_parts = []
        for doc_id, res in results.items():
            dm = self._collection.get(doc_id)
            context_parts.append(
                f"From '{dm.document_name}':\n{res['answer']}"
            )
        combined = "\n\n---\n\n".join(context_parts)

        response = self._llm.invoke([
            HumanMessage(content=(
                f"You retrieved information from {len(results)} documents.\n\n"
                f"{combined}\n\n"
                f"Original question: {query}\n\n"
                "Synthesize a single, comprehensive answer. "
                "If documents conflict, state both perspectives."
            ))
        ])
        return {
            "answer": _content_to_str(response.content),
            "documents_searched": list(results.keys()),
            "per_doc_results": results,
        }
```

### Building the Meta-Index During Ingestion

```python
def _build_document_abstract(self, dm: DocumentManager) -> str:
    """Generate a corpus-level abstract for one document."""
    response = self._llm.invoke([
        SystemMessage(content=(
            "Produce a 3-5 sentence abstract of this document. "
            "Include specific numbers, dates, key topics, and named entities. "
            "Focus on what questions this document can answer."
        )),
        HumanMessage(content=(
            f"Document tree:\n{dm.tree_text}\n\n"
            "Write the abstract now."
        )),
    ])
    return _content_to_str(response.content)
```

### Pros and Cons

| ✅ Pros | ❌ Cons |
|--------|--------|
| Scales to 20+ documents cleanly | Requires a meta-index build step per document added |
| Context window per LLM call stays small | 3 Gemini calls minimum (meta-nav + doc-nav + synthesis) |
| Each document's navigation is full-fidelity PageIndex | Sequential by default (parallel needs extra work) |
| Adding a new document only adds one meta-index entry | Abstract quality affects routing accuracy |
| Natural explainability: "searched these docs, found these nodes" | |

### Best For

- **5–50 documents** across diverse topics
- Use cases where documents cover different domains (reports + contracts + manuals)
- Production deployments where context cost matters
- When you need clear explainability of *which* document was used

---

## 5. Approach 3 — Parallel Agentic Chains (Powerful)

### Core Idea

Run a full independent `AgenticRAGChain` on every document simultaneously using `ThreadPoolExecutor`. Each chain gets the same query and navigates its own tree independently. A final synthesis call merges all answers.

### Code Structure

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelMultiDocChain(Runnable):
    """
    Runs one full AgenticRAGChain per document in parallel threads.
    Merges results in a final Gemini synthesis call.
    """

    def __init__(self, collection: DocumentCollection, max_workers: int = 4):
        self._collection = collection
        self._max_workers = max_workers
        self._llm = ChatGoogleGenerativeAI(...)

    def invoke(self, input: dict, config=None) -> dict:
        query = input["query"]
        results = self._run_parallel(query)
        return self._synthesize(query, results)

    def _run_parallel(self, query: str) -> dict[str, dict]:
        results = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            future_to_doc = {
                ex.submit(self._query_single_doc, doc_id, query): doc_id
                for doc_id in self._collection.doc_ids()
            }
            for future in as_completed(future_to_doc):
                doc_id = future_to_doc[future]
                try:
                    results[doc_id] = future.result()
                except Exception as e:
                    results[doc_id] = {"answer": f"Error: {e}", "error": True}
        return results

    def _query_single_doc(self, doc_id: str, query: str) -> dict:
        dm     = self._collection.get(doc_id)
        chain  = AgenticRAGChain(dm)
        result = chain.invoke({"query": query})
        result["document_name"] = dm.document_name
        return result

    def _synthesize(self, query: str, results: dict) -> dict:
        # Filter out docs that had no useful answer
        useful = {
            doc_id: res for doc_id, res in results.items()
            if not res.get("error") and res.get("answer")
        }
        if not useful:
            return {"answer": "No relevant information found across documents."}

        context_parts = []
        all_pages, all_nodes, all_steps = [], [], []

        for doc_id, res in useful.items():
            dm = self._collection.get(doc_id)
            context_parts.append(
                f"SOURCE: {dm.document_name}\n"
                f"ANSWER: {res['answer']}\n"
                f"NODES: {', '.join(res.get('nodes_visited', []))}"
            )
            all_pages.extend(res.get("pages_accessed", []))
            all_nodes.extend(res.get("nodes_visited", []))
            all_steps.extend(res.get("steps", []))

        synthesis_prompt = (
            f"You have retrieved information from {len(useful)} documents "
            f"to answer this question: {query}\n\n"
            + "\n\n---\n\n".join(context_parts) +
            "\n\nSynthesize a single comprehensive answer. "
            "Cite which document each piece of information comes from. "
            "If documents contradict each other, state both positions clearly."
        )
        response = self._llm.invoke([HumanMessage(content=synthesis_prompt)])

        return {
            "answer": _content_to_str(response.content),
            "documents_searched": list(useful.keys()),
            "pages_accessed": sorted(set(all_pages)),
            "nodes_visited": all_nodes,
            "steps": all_steps,
            "per_doc_results": useful,
        }
```

### Execution Timeline

```
Query arrives
    │
    ├──── Thread 1: AgenticRAGChain(doc_1) ────► result_1  ─┐
    ├──── Thread 2: AgenticRAGChain(doc_2) ────► result_2  ─┤
    ├──── Thread 3: AgenticRAGChain(doc_3) ────► result_3  ─┤── Synthesize ──► Final Answer
    └──── Thread 4: AgenticRAGChain(doc_4) ────► result_4  ─┘

Wall time ≈ max(single_doc_time) + synthesis_time
        NOT sum(all_doc_times)
```

### Pros and Cons

| ✅ Pros | ❌ Cons |
|--------|--------|
| Each document gets full reasoning power | N documents × up to 5 Gemini calls = expensive |
| Parallel execution — wall time ≈ slowest single doc | API rate limits hit quickly at scale |
| No meta-index build step required | All documents searched even if irrelevant |
| Full per-document reasoning trail preserved | |
| Most accurate — no navigation shortcuts | |

### Best For

- **2–8 documents** where all are potentially relevant
- High-stakes queries where accuracy matters more than cost
- Regulatory / compliance use cases (search everything, miss nothing)
- When documents are tightly interrelated and you can't predict which is relevant

---

## 6. The Synthesis Problem

All three approaches converge on the same hard problem: **what do you do when documents disagree?**

A naive synthesis just concatenates answers. A good synthesis reasons about conflicts.

```python
_SYNTHESIS_SYSTEM = """\
You are a document synthesis agent. You have retrieved answers from multiple
source documents. Your job is to produce ONE final answer that:

1. Integrates information from all sources coherently
2. Explicitly cites which document each fact comes from
3. Flags any contradictions between documents and presents both sides
4. Ranks information by recency or authority when sources conflict
5. States clearly if the question cannot be fully answered from the available documents

Be precise. Be honest about uncertainty. Never fabricate figures.
"""
```

### Types of Cross-Document Conflicts

| Conflict Type | Example | Synthesis Strategy |
|--------------|---------|-------------------|
| **Factual disagreement** | Doc A: revenue $4.2B, Doc B: revenue $4.1B | Report both, note the discrepancy, suggest checking filing date |
| **Temporal mismatch** | Doc A: 2023 data, Doc B: 2024 data | Prefer most recent, note the comparison is across periods |
| **Scope mismatch** | Doc A: global revenue, Doc B: APAC only | Clarify scope before comparing |
| **Definitional difference** | Doc A: "gross revenue", Doc B: "net revenue" | Flag definitional difference explicitly |

---

## 7. Comparison Matrix

| Dimension | Approach 1 (Concatenated) | Approach 2 (Meta-Tree) | Approach 3 (Parallel) |
|-----------|--------------------------|----------------------|----------------------|
| **Max documents** | 2–5 | 5–50 | 2–8 |
| **Gemini calls per query** | 1–5 (loop) | 3–10 (meta + per-doc + synthesis) | N×(1–5) + 1 |
| **Context window pressure** | High (all trees combined) | Low (meta-index only at L1) | Low (one tree per call) |
| **Implementation effort** | Low | Medium | Medium |
| **Cross-doc reasoning** | In one pass | Sequential with synthesis | Parallel with synthesis |
| **Handles irrelevant docs** | ❌ (all docs in context) | ✅ (meta-nav filters) | ❌ (searches all) |
| **Explainability** | Medium | High | Very High |
| **Accuracy (structured docs)** | Good | Very Good | Best |
| **API cost** | Low | Medium | High |
| **Conflict detection** | Limited | Good | Best |

---

## 8. Recommended Migration Path

### Phase 1 — Foundation (applies to all approaches)

These changes are required regardless of which approach you choose:

```
1. Replace DocumentManager singleton with DocumentCollection
2. Namespace all node IDs as {doc_id}:{node_id}
3. Update find_node() and get_pages() to accept doc_id
4. Update /api/upload to append rather than replace
5. Update /api/status to return document list
6. Add /api/documents endpoint (list, delete)
```

### Phase 2 — Choose Your Approach

```
Small corpus (≤5 docs, related topics)    →  Approach 1
Production system (5-50 docs)             →  Approach 2  ← recommended default
High-stakes / compliance (≤8 docs)        →  Approach 3
```

### Phase 3 — Synthesis Quality

```
7. Write and test the synthesis prompt with real conflict cases
8. Add conflict detection logic (flag when docs disagree on figures)
9. Add source attribution to every synthesized claim
10. Add a "document relevance score" to filter low-signal results before synthesis
```

---

## Summary

PageIndex's tree structure makes multi-document scaling more tractable than chunk-based RAG, because each document's index is **compact, semantically rich, and independently navigable**. The architecture is fundamentally hierarchical — adding a meta-level above the per-document trees is a natural extension, not a redesign.

The three approaches are not mutually exclusive. A production system might use **Approach 2 as the default** (meta-tree routing is fast and cost-efficient) and **fall back to Approach 3** (parallel full search) when the meta-navigator is uncertain about which documents to select.

The hardest engineering problem is not navigation — it is **synthesis**: building a final answer that is honest about what each document says, explicit about conflicts, and precise about sources. That is where the real accuracy work happens.

---

*Document generated for the Vectorless RAG project — PageIndex multi-document extension design.*
