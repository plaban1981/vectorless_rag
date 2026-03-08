"""
Agentic Text RAG Chain
───────────────────────
Multi-hop reasoning loop: the LLM navigates the document tree by iteratively
fetching nodes until it has enough context to answer.

Each iteration Gemini either:
  • FETCH_NODE — requests the full text of a specific tree node
  • ANSWER     — returns the final answer with sources

Implemented as a LangChain Runnable for composability in LCEL pipelines.
The conversation history is maintained as proper LangChain message objects,
giving full visibility into the reasoning chain.
"""
import json
import logging
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import settings
from ..document_manager import DocumentManager

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert document analyst navigating a document using a hierarchical tree index.

Your task: Answer the user's question by strategically navigating to the right sections.

You have access to a TREE INDEX showing the document's structure with dense summaries.
At each step you will either:
  1. REQUEST a node's full content (to read the actual text)
  2. Provide your FINAL ANSWER (once you have sufficient information)

RULES:
- Read the tree index carefully. Use summaries to decide where to look first.
- Think step by step. Explain your reasoning before each action.
- Follow cross-references mentioned in summaries (e.g. "see N007", "refer to Appendix A").
- You may request multiple nodes across iterations to build a comprehensive answer.
- Only provide ANSWER when you are confident you have enough information.

RESPONSE FORMAT — strict JSON only, no markdown fences, no extra text:

To request a node:
{"action": "FETCH_NODE", "node_id": "N008", "reasoning": "The question asks about X. Node N008 covers Y which is relevant because Z."}

To give the final answer:
{"action": "ANSWER", "answer": "Based on the document...", "reasoning": "I gathered information from nodes X and Y which together provide...", "sources": ["N008 (pages 13-15)", "N013 (page 23)"]}

CRITICAL: Respond with ONLY valid JSON. No markdown. No text outside the JSON object."""


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

def _parse_llm_json(raw: str) -> dict:
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
    # Fallback: return the raw text as a direct answer
    return {"action": "ANSWER", "answer": raw, "reasoning": "JSON parse fallback", "sources": []}


# ── AgenticRAGChain ────────────────────────────────────────────────────────────

class AgenticRAGChain(Runnable):
    """
    Multi-hop agentic text RAG chain.

    Input:  {"query": str}
    Output: {
        "answer": str,
        "steps": list[dict],          # full reasoning trail
        "nodes_visited": list[str],
        "pages_accessed": list[int],
        "num_iterations": int,
        "timing": {"total_ms": int, "per_step_ms": list[int]},
        "sources": list[str],
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
        if not self._doc.pages:
            return {"answer": "No document loaded.", "error": True}
        return self._run_loop(input["query"])

    # ── Core agentic loop ──────────────────────────────────────────────────────

    def _run_loop(self, query: str) -> dict:
        t0 = time.time()

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Here is the document tree index:\n\n{self._doc.tree_text}\n\n"
                f"QUESTION: {query}\n\n"
                "Navigate the tree to find the answer. Start by reasoning about "
                "which node(s) to check first.\nRespond with ONLY valid JSON."
            )),
        ]

        steps: list[dict] = []
        pages_accessed: set[int] = set()
        nodes_visited: list[str] = []
        iteration = 0

        while iteration < settings.max_iterations:
            iteration += 1
            t_step = time.time()

            response = self._llm.invoke(messages)
            raw = _content_to_str(response.content)
            elapsed_ms = round((time.time() - t_step) * 1000)

            decision = _parse_llm_json(raw)
            action = decision.get("action", "ANSWER")
            reasoning = decision.get("reasoning", "")

            if action == "FETCH_NODE":
                node_id = decision.get("node_id", "")
                node = self._doc.find_node(node_id)

                if node:
                    content = self._doc.get_pages(node["start_page"], node["end_page"])
                    for p in range(node["start_page"], node["end_page"] + 1):
                        pages_accessed.add(p)
                    nodes_visited.append(node_id)

                    steps.append({
                        "step": iteration,
                        "action": "FETCH_NODE",
                        "node_id": node_id,
                        "node_title": node["title"],
                        "pages": f"{node['start_page']}-{node['end_page']}",
                        "reasoning": reasoning,
                        "duration_ms": elapsed_ms,
                    })
                    messages.append(AIMessage(content=raw))
                    messages.append(HumanMessage(content=(
                        f"Content of node {node_id} "
                        f"({node['title']}, pages {node['start_page']}-{node['end_page']}):\n\n"
                        f"{content}\n\n"
                        "Based on this and any previously fetched content, either:\n"
                        "- Request another node if you need more information\n"
                        "- Provide your FINAL ANSWER if you have enough\n\n"
                        f"Original question: {query}\nRespond with ONLY valid JSON."
                    )))

                else:
                    steps.append({
                        "step": iteration,
                        "action": "ERROR",
                        "detail": f"Node {node_id} not found in tree",
                        "reasoning": reasoning,
                        "duration_ms": elapsed_ms,
                    })
                    messages.append(AIMessage(content=raw))
                    messages.append(HumanMessage(content=(
                        f"Node '{node_id}' was not found. "
                        "Please check the tree index and request a valid node_id. "
                        "Respond with ONLY valid JSON."
                    )))

            elif action == "ANSWER":
                sources = decision.get("sources", [])
                steps.append({
                    "step": iteration,
                    "action": "ANSWER",
                    "reasoning": reasoning,
                    "sources": sources,
                    "duration_ms": elapsed_ms,
                })
                return {
                    "answer": decision.get("answer", ""),
                    "steps": steps,
                    "nodes_visited": nodes_visited,
                    "pages_accessed": sorted(pages_accessed),
                    "num_iterations": iteration,
                    "timing": {
                        "total_ms": round((time.time() - t0) * 1000),
                        "per_step_ms": [s.get("duration_ms", 0) for s in steps],
                    },
                    "sources": sources,
                }

        # Max iterations reached — force a final answer
        messages.append(HumanMessage(content=(
            f"Maximum iterations reached. Provide your best answer to: {query}\n"
            "Respond with ONLY valid JSON with action ANSWER."
        )))
        response = self._llm.invoke(messages)
        raw_final = _content_to_str(response.content)
        decision = _parse_llm_json(raw_final)
        steps.append({
            "step": iteration + 1,
            "action": "FORCED_ANSWER",
            "reasoning": "Max iterations reached",
            "duration_ms": 0,
        })
        return {
            "answer": decision.get("answer", raw_final),
            "steps": steps,
            "nodes_visited": nodes_visited,
            "pages_accessed": sorted(pages_accessed),
            "num_iterations": iteration + 1,
            "timing": {
                "total_ms": round((time.time() - t0) * 1000),
                "per_step_ms": [s.get("duration_ms", 0) for s in steps],
            },
            "sources": decision.get("sources", []),
            "note": "Answer may be incomplete — max reasoning iterations reached",
        }
