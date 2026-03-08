"""
Vectorless Multimodal RAG — FastAPI Backend
════════════════════════════════════════════
Start:  uv run uvicorn app:app --reload --port 8000
Docs:   http://localhost:8000/docs
"""
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vectorless_rag.pipeline import pipeline

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vectorless Multimodal RAG",
    description=(
        "PageIndex hierarchical tree navigation + Gemini multimodal LLM + LangChain. "
        "No vectors. No chunking. No OCR. Pure reasoning."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)
FRONTEND = Path(__file__).parent / "frontend" / "index.html"

# ── Request/Response models ────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    mode: str = "text"  # "text" | "vision"


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def serve_ui():
    if FRONTEND.exists():
        return FileResponse(str(FRONTEND))
    return {"status": "ok", "message": "Vectorless RAG API — see /docs"}


@app.get("/api/status")
async def status():
    """Return current document info and readiness."""
    return {
        "document_name": pipeline.document_name,
        "total_pages": pipeline.total_pages,
        "has_document": pipeline.has_document,
        "tree_nodes": pipeline.tree_node_count,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF document.

    Triggers:
      1. Page text extraction (PyMuPDF)
      2. Tree index generation (Gemini)  ← may take 20-60s for large docs
      3. Page image rendering (PyMuPDF)

    Returns metadata dict with timing breakdown.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large — max 50 MB.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, pipeline.load_document, file_bytes, file.filename
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.get("/api/tree")
async def get_tree():
    """Return the full hierarchical tree index for the loaded document."""
    if not pipeline.has_document:
        raise HTTPException(status_code=400, detail="No document loaded.")
    return pipeline._doc.tree_index


@app.post("/api/query")
async def query(req: QueryRequest):
    """
    Run a query against the loaded document.

    ``mode="text"``   — agentic multi-hop tree navigation (reads text nodes)
    ``mode="vision"`` — tree navigation + Gemini VLM on page images (no OCR)
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if req.mode not in ("text", "vision"):
        raise HTTPException(status_code=400, detail="mode must be 'text' or 'vision'.")
    if not pipeline.has_document:
        raise HTTPException(status_code=400, detail="No document loaded — upload a PDF first.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, pipeline.query, req.query, req.mode)
        return result
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
