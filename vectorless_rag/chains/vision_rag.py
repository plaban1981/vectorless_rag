"""
Vision RAG Chain
─────────────────
Bypasses OCR entirely — Gemini sees raw PDF page images directly.

Pipeline (two LLM calls total):
  1. Tree navigation  — Gemini reads the tree index (text) and identifies which
                         nodes are relevant, returning their node_ids.
  2. Visual answering — Gemini receives the actual JPEG images of those pages
                         and produces the final answer from visual context.

This preserves tables, charts, column layouts, and visual structure that
standard OCR destroys.

Implemented as a LangChain Runnable.
"""
import json
import logging
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import settings
from ..document_manager import DocumentManager

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

_NAV_SYSTEM = """\
You are a precise document navigator. Given a question and a hierarchical tree index,
identify which nodes most likely contain the answer.

Return ONLY valid JSON (no markdown, no extra text):
{"reasoning": "brief explanation", "node_ids": ["N001", "N002"]}

Choose the 1-4 most relevant nodes. Prefer specific leaf nodes over broad parent nodes."""

_VISION_PROMPT = """\
You are a precise document analyst with full visual understanding.
You are looking at page images from a document.

Answer the question based ONLY on what you can see in the provided pages.

Guidelines:
- Preserve exact numbers, figures, percentages, and table data
- Note which page/section each piece of information comes from
- If the answer spans multiple pages, synthesize them coherently
- If the answer is not visible in these pages, state that clearly

Pages shown: {pages}

Question: {query}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _content_to_str(content) -> str:
    """Normalise LangChain response.content — may be str or list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


# ── JSON parsing ───────────────────────────────────────────────────────────────

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
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"reasoning": "parse error", "node_ids": []}


# ── VisionRAGChain ─────────────────────────────────────────────────────────────

class VisionRAGChain(Runnable):
    """
    Vision-based RAG chain: tree navigation → page images → Gemini VLM answer.

    Input:  {"query": str}
    Output: {
        "answer": str,
        "nodes_visited": list[str],
        "pages_accessed": list[int],
        "navigation_reasoning": str,
        "timing": {"total_ms", "navigation_ms", "image_load_ms", "vision_ms"},
    }
    """

    def __init__(self, doc_manager: DocumentManager) -> None:
        self._doc = doc_manager
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=settings.temperature,
            google_api_key=settings.google_api_key,
        )

    def invoke(self, input: dict[str, Any], config: Optional[RunnableConfig] = None) -> dict:
        if not self._doc.tree_index:
            return {"answer": "No document loaded.", "error": True}
        return self._run(input["query"])

    # ── Pipeline ───────────────────────────────────────────────────────────────

    def _run(self, query: str) -> dict:
        t0 = time.time()

        # Step 1: Navigate tree → identify relevant node_ids
        t1 = time.time()
        node_ids, nav_reasoning = self._navigate_tree(query)
        t_nav = time.time() - t1

        if not node_ids:
            return {
                "answer": "Could not identify relevant sections in the document tree.",
                "nodes_visited": [],
                "pages_accessed": [],
                "navigation_reasoning": nav_reasoning,
                "timing": {"total_ms": round((time.time() - t0) * 1000), "navigation_ms": round(t_nav * 1000)},
            }

        # Step 2: Collect unique pages for those nodes
        pages_set: set[int] = set()
        nodes_visited: list[str] = []
        for node_id in node_ids:
            node = self._doc.find_node(node_id)
            if node:
                for p in range(node["start_page"], node["end_page"] + 1):
                    pages_set.add(p)
                nodes_visited.append(node_id)
        sorted_pages = sorted(pages_set)
        logger.info(f"Vision RAG: nodes={nodes_visited} → pages={sorted_pages}")

        # Step 3: Load page images from disk
        t2 = time.time()
        images_b64 = self._doc.get_page_images_b64(sorted_pages)
        t_images = time.time() - t2

        if not images_b64:
            return {
                "answer": "Page images not available for the identified sections.",
                "nodes_visited": nodes_visited,
                "pages_accessed": sorted_pages,
                "navigation_reasoning": nav_reasoning,
                "timing": {"total_ms": round((time.time() - t0) * 1000)},
            }

        # Step 4: Answer with Gemini vision
        t3 = time.time()
        answer = self._answer_with_vision(query, images_b64, sorted_pages)
        t_vision = time.time() - t3

        return {
            "answer": answer,
            "nodes_visited": nodes_visited,
            "pages_accessed": sorted_pages,
            "navigation_reasoning": nav_reasoning,
            "timing": {
                "total_ms": round((time.time() - t0) * 1000),
                "navigation_ms": round(t_nav * 1000),
                "image_load_ms": round(t_images * 1000),
                "vision_ms": round(t_vision * 1000),
            },
        }

    def _navigate_tree(self, query: str) -> tuple[list[str], str]:
        """One-shot tree navigation: identify relevant node_ids in a single LLM call."""
        prompt = (
            f"Question: {query}\n\n"
            f"Document tree:\n{self._doc.tree_text}\n\n"
            'Return JSON: {"reasoning": "...", "node_ids": ["N001"]}'
        )
        response = self._llm.invoke([
            SystemMessage(content=_NAV_SYSTEM),
            HumanMessage(content=prompt),
        ])
        result = _parse_json(_content_to_str(response.content))
        node_ids = result.get("node_ids", [])
        reasoning = result.get("reasoning", "")
        logger.info(f"Tree nav → {node_ids} | {reasoning[:100]}")
        return node_ids, reasoning

    def _answer_with_vision(self, query: str, images_b64: list[str], page_nums: list[int]) -> str:
        """Build a multimodal message (text + images) and call Gemini."""
        content: list[dict] = [{
            "type": "text",
            "text": _VISION_PROMPT.format(pages=page_nums, query=query),
        }]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        response = self._llm.invoke([HumanMessage(content=content)])
        return _content_to_str(response.content)
