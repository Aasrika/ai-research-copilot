"""
session_manager.py
-------------------
Session isolation layer for Mode 1 (Literature Assistant) and Mode 2
(Draft Critique). Each session gets its own on-disk directory tree and
its own FAISS index, so papers uploaded into one session can never be
retrieved from another.

WHY THE DRAFT PAPER IS NEVER INDEXED:
None of the Mode 2 critique features (section-presence check, claim
extraction, claim-literature alignment) need the draft to be semantically
searchable — section-check and claim-extraction work off its raw parsed
text, and claim-alignment only ever retrieves from the *literature*
store. So the draft is just saved to disk per-session and parsed on
demand. This keeps literature retrieval structurally incapable of
surfacing draft content, without any extra metadata filtering.

Manifest is a flat JSON file (data/sessions/sessions.json), matching the
project's existing lightweight-file-persistence convention (see
evaluation/runs.jsonl) rather than adding a database dependency.
"""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS

from core.config import SESSIONS_DIR
from core.document_processor import build_or_update_index, load_index

MANIFEST_PATH = SESSIONS_DIR / "sessions.json"


# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST I/O
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def _literature_dir(session_id: str) -> Path:
    return _session_dir(session_id) / "literature"


def _draft_dir(session_id: str) -> Path:
    return _session_dir(session_id) / "draft"


def _index_dir(session_id: str) -> Path:
    return _session_dir(session_id) / "index"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_session(name: str) -> dict:
    session_id = uuid.uuid4().hex[:8]

    _literature_dir(session_id).mkdir(parents=True, exist_ok=True)
    _draft_dir(session_id).mkdir(parents=True, exist_ok=True)
    _index_dir(session_id).mkdir(parents=True, exist_ok=True)

    record = {
        "id": session_id,
        "name": name.strip() or f"Session {session_id}",
        "created_at": _now(),
        "updated_at": _now(),
        "literature_papers": [],
        "draft_paper": None,
    }

    manifest = _load_manifest()
    manifest[session_id] = record
    _save_manifest(manifest)

    return record


def list_sessions() -> list[dict]:
    manifest = _load_manifest()
    return sorted(manifest.values(), key=lambda s: s["updated_at"], reverse=True)


def get_session(session_id: str) -> Optional[dict]:
    return _load_manifest().get(session_id)


def delete_session(session_id: str) -> None:
    """
    Removes the manifest entry and the session's entire on-disk directory
    (literature/, draft/, index/). Any session-scoped Streamlit
    session_state (chat history, cached critique reports, etc.) is the
    caller's responsibility to clean up — this module is UI-agnostic.
    """
    manifest = _load_manifest()
    manifest.pop(session_id, None)
    _save_manifest(manifest)
    shutil.rmtree(_session_dir(session_id), ignore_errors=True)


def _touch(session_id: str, **updates) -> dict:
    manifest = _load_manifest()
    record = manifest[session_id]
    record.update(updates)
    record["updated_at"] = _now()
    manifest[session_id] = record
    _save_manifest(manifest)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# LITERATURE (indexed, session-isolated)
# ─────────────────────────────────────────────────────────────────────────────

def add_literature(session_id: str, files: list[tuple[str, bytes]]) -> FAISS:
    """
    files: list of (filename, pdf_bytes) — e.g. from Streamlit's file_uploader.
    Saves each file under this session's literature/ dir, then builds/updates
    this session's own FAISS index (never the global one).
    """
    lit_dir = _literature_dir(session_id)
    lit_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for filename, content in files:
        dest = lit_dir / filename
        dest.write_bytes(content)
        paths.append(str(dest))

    store = build_or_update_index(paths, index_dir=str(_index_dir(session_id)))

    session = get_session(session_id)
    existing = set(session.get("literature_papers", [])) if session else set()
    existing.update(f.name for f in lit_dir.glob("*.pdf"))
    _touch(session_id, literature_papers=sorted(existing))

    return store


def load_literature_store(session_id: str) -> Optional[FAISS]:
    index_dir = _index_dir(session_id)
    if not (index_dir / "index.faiss").exists():
        return None
    return load_index(index_dir=str(index_dir))


def get_indexed_literature(session_id: str) -> list[str]:
    """
    Distinct `paper_title` values (filename stems) actually present in this
    session's FAISS index — this is what `retrieve()`'s paper_filter matches
    against, so the UI's paper picker must show the same stems, not raw
    uploaded filenames.
    """
    store = load_literature_store(session_id)
    if store is None:
        return []
    titles = {
        doc.metadata.get("paper_title", "Unknown")
        for doc in store.docstore._dict.values()
    }
    return sorted(titles)


# ─────────────────────────────────────────────────────────────────────────────
# DRAFT (never indexed — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

def set_draft(session_id: str, filename: str, content: bytes) -> Path:
    draft_dir = _draft_dir(session_id)
    draft_dir.mkdir(parents=True, exist_ok=True)

    # Only one draft per session — clear any previous draft file first.
    for old in draft_dir.glob("*.pdf"):
        old.unlink()

    dest = draft_dir / filename
    dest.write_bytes(content)

    _touch(session_id, draft_paper=filename)

    return dest


def get_draft_path(session_id: str) -> Optional[Path]:
    session = get_session(session_id)
    if not session or not session.get("draft_paper"):
        return None
    path = _draft_dir(session_id) / session["draft_paper"]
    return path if path.exists() else None
