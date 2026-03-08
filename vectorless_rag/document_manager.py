"""
Document Manager
────────────────
Handles the full document ingestion pipeline:
  1. PDF text extraction (PyMuPDF)
  2. Hierarchical tree index generation (Gemini via LangChain)
  3. PDF page image rendering (PyMuPDF — for vision mode)
  4. Disk caching keyed by file SHA-256 hash

Both text and vision RAG chains share this single manager instance.
"""
import base64
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import settings

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

_TREE_SYSTEM = (
    "You are an expert document analyst. Create comprehensive hierarchical JSON tree "
    "indexes for documents that will be used by an LLM to navigate and retrieve "
    "information at query time. Always respond with valid JSON only — "
    "no markdown fences, no preamble, no trailing text."
)

_TREE_USER = """\
Analyze this document and create a hierarchical JSON tree index.

Requirements:
1. Each node must have: node_id (N001, N002…), title, start_page, end_page, summary
2. Follow the document's natural section structure (chapters → sections → subsections)
3. Summaries must be DENSE: include specific numbers, dates, names, and cross-references
4. Add sub_nodes for any section spanning more than 3 pages
5. Capture every major content area — do not skip appendices or tables

Output this exact JSON structure (no markdown, no extra text):
{{
  "document": {{
    "title": "...",
    "description": "One comprehensive paragraph describing the document and its key topics",
    "total_pages": {total_pages},
    "nodes": [
      {{
        "node_id": "N001",
        "title": "...",
        "start_page": 1,
        "end_page": 2,
        "summary": "Dense summary with specific facts, figures, cross-references...",
        "sub_nodes": []
      }}
    ]
  }}
}}

DOCUMENT ({total_pages} pages):
{doc_text}

Respond with ONLY valid JSON."""


# ── DocumentManager ────────────────────────────────────────────────────────────

class DocumentManager:
    """
    Thread-safe document state container.

    Attributes:
        document_name   Original filename
        pages           {page_num: extracted_text} — 1-indexed
        total_pages     Physical page count (including blank pages)
        tree_index      Parsed JSON tree {"document": {...}}
        tree_text       Human-readable formatted tree for LLM context
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.document_name: str = ""
        self.pages: dict[int, str] = {}
        self.total_pages: int = 0
        self.tree_index: Optional[dict] = None
        self.tree_text: str = ""
        self._page_image_paths: dict[int, Path] = {}
        self._file_hash: str = ""

        # Ensure cache directories exist
        settings.trees_dir.mkdir(parents=True, exist_ok=True)
        settings.images_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_pdf(self, file_bytes: bytes, filename: str) -> dict:
        """
        Full ingestion pipeline — call once per document.

        Returns metadata dict with timing breakdown.
        Raises ValueError for unreadable PDFs.
        """
        with self._lock:
            t0 = time.time()
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
            self.document_name = filename
            self._file_hash = file_hash

            # 1. Extract text per page
            t1 = time.time()
            self._extract_text(file_bytes)
            t_extract = time.time() - t1

            # 2. Build or load cached tree
            t2 = time.time()
            cache_path = settings.trees_dir / f"{file_hash}.json"
            if cache_path.exists():
                logger.info(f"Loading cached tree: {cache_path}")
                with open(cache_path) as f:
                    self.tree_index = json.load(f)
            else:
                logger.info("Building tree index with Gemini...")
                self.tree_index = self._build_tree_with_gemini()
                with open(cache_path, "w") as f:
                    json.dump(self.tree_index, f, indent=2)
            self.tree_text = self._format_tree(self.tree_index["document"])
            t_tree = time.time() - t2

            # 3. Render page images (cached per page — skip if already on disk)
            t3 = time.time()
            self._render_page_images(file_bytes, file_hash)
            t_images = time.time() - t3

            tree_nodes = len(self.tree_index["document"]["nodes"])
            logger.info(
                f"Loaded '{filename}' | {self.total_pages} pages | "
                f"{tree_nodes} nodes | {(time.time()-t0):.1f}s"
            )
            return {
                "document_name": filename,
                "total_pages": self.total_pages,
                "text_pages": len(self.pages),
                "tree_nodes": tree_nodes,
                "timing": {
                    "extract_ms": round(t_extract * 1000),
                    "tree_ms": round(t_tree * 1000),
                    "images_ms": round(t_images * 1000),
                    "total_ms": round((time.time() - t0) * 1000),
                },
            }

    def get_pages(self, start: int, end: int) -> str:
        """Return extracted text for page range [start, end] inclusive."""
        return "\n\n".join(
            f"[PAGE {p}]\n{self.pages[p]}"
            for p in range(start, end + 1)
            if p in self.pages
        )

    def get_page_images_b64(self, page_nums: list[int]) -> list[str]:
        """Return base64-encoded JPEG images for the given page numbers."""
        result = []
        for p in page_nums:
            if p in self._page_image_paths:
                with open(self._page_image_paths[p], "rb") as f:
                    result.append(base64.b64encode(f.read()).decode("utf-8"))
        return result

    def find_node(self, node_id: str) -> Optional[dict]:
        """Traverse tree to find a node by its ID (recursive)."""
        if self.tree_index is None:
            return None
        return self._find_node_recursive(node_id, self.tree_index["document"]["nodes"])

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_text(self, file_bytes: bytes) -> None:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        self.total_pages = len(doc)
        self.pages = {}
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                self.pages[i + 1] = text
        doc.close()
        if not self.pages:
            raise ValueError(
                "No text could be extracted from this PDF. It may be scanned/image-only. "
                "Vision mode will still work — select 'Vision RAG' in the UI."
            )

    def _render_page_images(self, file_bytes: bytes, file_hash: str) -> None:
        img_dir = settings.images_dir / file_hash
        img_dir.mkdir(parents=True, exist_ok=True)
        mat = fitz.Matrix(settings.image_scale, settings.image_scale)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        self._page_image_paths = {}
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            img_path = img_dir / f"page_{page_num}.jpg"
            if not img_path.exists():
                pix = doc.load_page(page_idx).get_pixmap(matrix=mat)
                pix.save(str(img_path))
            self._page_image_paths[page_num] = img_path
        doc.close()

    def _build_tree_with_gemini(self) -> dict:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.1,  # Slightly warmer for richer summaries
            google_api_key=settings.google_api_key,
        )
        # Send first 50 pages; truncate if too large for context
        pages_to_send = dict(list(sorted(self.pages.items()))[:50])
        doc_text = "\n\n".join(
            f"[PAGE {p}]\n{content}" for p, content in sorted(pages_to_send.items())
        )
        if len(doc_text) > 100_000:
            doc_text = doc_text[:100_000] + "\n\n[… document continues …]"

        response = llm.invoke([
            SystemMessage(content=_TREE_SYSTEM),
            HumanMessage(content=_TREE_USER.format(
                total_pages=self.total_pages, doc_text=doc_text
            )),
        ])
        content = response.content
        # Newer langchain-google-genai versions can return a list of content blocks
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        logger.debug("Raw Gemini tree response (first 500 chars): %s", content[:500])
        try:
            return self._parse_json(content)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to parse tree JSON. Raw response:\n%s", content[:2000])
            raise

    def _format_tree(self, doc: dict) -> str:
        lines = [
            f"DOCUMENT: {doc['title']}",
            f"Description: {doc.get('description', '')}",
            f"Total pages: {doc.get('total_pages', self.total_pages)}",
            "",
            "TREE INDEX (node_id | title | pages | summary):",
            "=" * 70,
        ]

        def fmt(node: dict, indent: int = 0) -> None:
            prefix = "  " * indent
            lines.append(
                f"{prefix}[{node['node_id']}] {node['title']} "
                f"(pages {node['start_page']}-{node['end_page']})"
            )
            lines.append(f"{prefix}  Summary: {node['summary']}")
            lines.append("")
            for child in node.get("sub_nodes", []):
                fmt(child, indent + 1)

        for node in doc.get("nodes", []):
            fmt(node)
        return "\n".join(lines)

    def _find_node_recursive(self, node_id: str, nodes: list) -> Optional[dict]:
        for node in nodes:
            if node["node_id"] == node_id:
                return node
            found = self._find_node_recursive(node_id, node.get("sub_nodes", []))
            if found:
                return found
        return None

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise ValueError(f"Could not parse JSON from LLM: {raw[:300]}")


# ── Global singleton (shared by chains and pipeline) ──────────────────────────
doc_manager = DocumentManager()
