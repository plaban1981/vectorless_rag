"""
VectorlessRAGPipeline
──────────────────────
Main orchestrator — wraps DocumentManager + both chains behind a single
clean interface used by both the CLI (main.py) and the API (app.py).
"""
from typing import Literal

from .document_manager import DocumentManager, doc_manager
from .chains.agentic_rag import AgenticRAGChain
from .chains.vision_rag import VisionRAGChain


class VectorlessRAGPipeline:
    """
    Vectorless Multimodal RAG pipeline.

    Supports two retrieval modes:
      text   — multi-hop agentic tree navigation; reads extracted text nodes
      vision — one-shot tree navigation + Gemini VLM on raw page images (no OCR)

    Usage::

        pipeline = VectorlessRAGPipeline()
        pipeline.load_document(file_bytes, "report.pdf")
        result = pipeline.query("What was Q3 revenue?", mode="vision")
    """

    def __init__(self, document_manager: DocumentManager = doc_manager) -> None:
        self._doc = document_manager
        self._text_chain = AgenticRAGChain(document_manager)
        self._vision_chain = VisionRAGChain(document_manager)

    def load_document(self, file_bytes: bytes, filename: str) -> dict:
        """Ingest a PDF. Returns metadata dict with timing breakdown."""
        return self._doc.load_pdf(file_bytes, filename)

    def query(self, question: str, mode: Literal["text", "vision"] = "text") -> dict:
        """
        Query the loaded document.

        Args:
            question: Natural language question about the document.
            mode: ``"text"`` for agentic text RAG; ``"vision"`` for page-image VLM RAG.

        Returns:
            dict containing ``answer``, ``pages_accessed``, ``nodes_visited``,
            ``timing``, and (for text mode) ``steps`` with the full reasoning trail.
        """
        if not self._doc.pages and not self._doc.tree_index:
            return {"answer": "No document loaded. Please upload a PDF first.", "error": True}

        if mode == "text":
            result = self._text_chain.invoke({"query": question})
        elif mode == "vision":
            result = self._vision_chain.invoke({"query": question})
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'text' or 'vision'.")

        result["query"] = question
        result["mode"] = mode
        return result

    @property
    def document_name(self) -> str:
        return self._doc.document_name

    @property
    def total_pages(self) -> int:
        return self._doc.total_pages

    @property
    def has_document(self) -> bool:
        return bool(self._doc.pages or self._doc.tree_index)

    @property
    def tree_node_count(self) -> int:
        if self._doc.tree_index:
            return len(self._doc.tree_index["document"]["nodes"])
        return 0


# Global singleton used by the FastAPI app
pipeline = VectorlessRAGPipeline()
