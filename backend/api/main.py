"""
main.py  (FastAPI backend)
--------------------------
Exposes the entire pipeline as a REST API.

WHY FASTAPI ON TOP OF STREAMLIT?
Streamlit is great for demos but it's a single-user, browser-only tool.
FastAPI makes your pipeline callable by:
  - Any frontend (React, mobile apps)
  - Other services (CI pipelines, notebooks, scripts)
  - Automated evaluation harnesses
  - Future: a real product with your own auth + billing

This is what separates "I built a demo" from "I built a service."

ENDPOINTS:
  POST /query         — run the full Q&A agent pipeline
  POST /compare       — run the comparison pipeline
  POST /ideas         — run the idea generation pipeline
  POST /upload        — upload + index a PDF
  GET  /papers        — list indexed papers
  GET  /metrics       — get evaluation metrics
  POST /feedback      — submit user rating for a run
  GET  /health        — health check (for Docker / deploy)

Run with:  uvicorn backend.api.main:app --reload --port 8000
"""

import sys, os, shutil, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from typing  import Optional
from contextlib import asynccontextmanager

from fastapi              import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic             import BaseModel
from dotenv               import load_dotenv

from core.document_processor import build_or_update_index, load_index, get_indexed_papers
from agents.graph             import run_pipeline
from agents.idea_agent        import generate_research_ideas
from agents.comparison_agent  import run_comparison
from evaluation.metrics       import compute_all_metrics
from evaluation.logger        import load_runs, update_feedback, RunRecord, log_run

load_dotenv()

# ── Global state — vector store loaded once at startup ────────────────────
_vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store when the API starts."""
    global _vector_store
    try:
        _vector_store = load_index()
        print("✅ Vector store loaded at startup")
    except Exception:
        print("⚠️  No index found at startup — upload papers to create one")
    yield   # app runs here
    # Cleanup on shutdown (nothing to do for FAISS)


app = FastAPI(
    title       = "AI Research Copilot API",
    description = "Multi-agent RAG pipeline for research paper analysis",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Allow Streamlit (and any dev frontend) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request/Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:        str
    paper_filter: Optional[str] = None
    k:            int           = 5
    max_retries:  int           = 2

class QueryResponse(BaseModel):
    run_id:              str
    query:               str
    answer:              str
    critic_score:        int
    verdict:             str
    hallucination_flags: list[str]
    retry_count:         int
    sources:             list[dict]
    latency_seconds:     float

class CompareRequest(BaseModel):
    paper_a:      str
    paper_b:      str
    aspect:       str = "Methodology"
    custom_query: str = ""

class IdeasRequest(BaseModel):
    paper_filter: Optional[str] = None
    focus_area:   str           = ""

class FeedbackRequest(BaseModel):
    run_id:  str
    rating:  int    # 1–5
    comment: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────

def _require_store():
    if _vector_store is None:
        raise HTTPException(503, "No index loaded. Upload papers first via POST /upload")
    return _vector_store


def _sources_from_chunks(chunks) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        key = (c.metadata["paper_title"], c.metadata["page"])
        if key not in seen:
            seen.add(key)
            out.append({"paper": c.metadata["paper_title"],
                        "page": c.metadata["page"],
                        "section": c.metadata["section"]})
    return out


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "index_loaded":  _vector_store is not None,
        "papers":        get_indexed_papers(),
    }


@app.get("/papers")
def list_papers():
    return {"papers": get_indexed_papers()}


@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Upload and index a PDF. Merges into existing index."""
    global _vector_store
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    papers_dir = Path("data/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)
    dest = papers_dir / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    _vector_store = build_or_update_index([str(dest)])
    return {
        "message":  f"'{file.filename}' indexed successfully.",
        "papers":   get_indexed_papers(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run the full multi-agent Q&A pipeline."""
    store = _require_store()
    t0    = time.perf_counter()

    state = run_pipeline(
        query        = req.query,
        vector_store = store,
        k            = req.k,
        paper_filter = req.paper_filter,
        max_retries  = req.max_retries,
    )

    return QueryResponse(
        run_id              = state.get("run_id", ""),
        query               = req.query,
        answer              = state["answer"],
        critic_score        = state["critic_score"],
        verdict             = state["verdict"],
        hallucination_flags = state["hallucination_flags"],
        retry_count         = state["retry_count"],
        sources             = _sources_from_chunks(state["chunks"]),
        latency_seconds     = round(time.perf_counter() - t0, 2),
    )


@app.post("/compare")
def compare(req: CompareRequest):
    """Run the structured paper comparison pipeline."""
    store  = _require_store()
    result = run_comparison(
        req.paper_a, req.paper_b, req.aspect,
        store, req.custom_query
    )
    # Log comparison run
    record = RunRecord(pipeline_type="comparison",
                       query=f"Compare {req.paper_a} vs {req.paper_b}: {req.aspect}",
                       num_chunks=len(result["chunks_a"]) + len(result["chunks_b"]))
    log_run(record)

    return {
        "paper_a":    req.paper_a,
        "paper_b":    req.paper_b,
        "aspect":     req.aspect,
        "comparison": result["structured"],
        "verdict":    result["verdict"],
    }


@app.post("/ideas")
def ideas(req: IdeasRequest):
    """Run the research idea generation pipeline."""
    store  = _require_store()
    result = generate_research_ideas(store, req.paper_filter, req.focus_area)
    # Log ideas run
    record = RunRecord(pipeline_type="ideas",
                       query=f"Ideas for {req.paper_filter or 'all'}: {req.focus_area}",
                       paper_filter=req.paper_filter,
                       num_chunks=len(result.get("chunks_used",[])))
    log_run(record)

    return {k: v for k, v in result.items() if k != "chunks_used"}


@app.get("/metrics")
def metrics():
    """Return all evaluation metrics."""
    return compute_all_metrics()


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Submit user rating for a specific run."""
    if not 1 <= req.rating <= 5:
        raise HTTPException(400, "Rating must be between 1 and 5")
    success = update_feedback(req.run_id, req.rating, req.comment)
    if not success:
        raise HTTPException(404, f"Run '{req.run_id}' not found")
    return {"message": "Feedback recorded.", "run_id": req.run_id}