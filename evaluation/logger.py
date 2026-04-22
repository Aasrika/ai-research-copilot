"""
logger.py
---------
Records every pipeline run to a JSON Lines file.

WHY JSON LINES (.jsonl)?
Each line is a valid JSON object — one per run.
Easy to append (no file locking issues), easy to parse,
human-readable, and git-diffable. No database needed for a project
of this scale — JSONL is the right tool here.

WHAT GETS LOGGED:
  - Timestamp, query, answer length
  - Critic score + verdict + hallucination flags
  - Retry count (key signal: how often does the system need to self-correct?)
  - Latency breakdown (retrieval vs generation vs critic)
  - Which paper(s) were searched
  - Pipeline type: qa | comparison | ideas

This data powers the evaluation dashboard and answers:
  "Is my system improving? Where does it fail? How fast is it?"
"""

import json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

LOG_PATH = Path("evaluation/runs.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class RunRecord:
    # Identity
    run_id:        str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:     str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_type: str   = "qa"          # qa | comparison | ideas

    # Query
    query:         str   = ""
    paper_filter:  Optional[str] = None
    k:             int   = 5

    # Output quality
    critic_score:        int        = 0
    verdict:             str        = ""
    hallucination_flags: list[str]  = field(default_factory=list)
    retry_count:         int        = 0
    answer_length:       int        = 0   # chars — proxy for answer completeness

    # Latency (seconds)
    latency_total:       float = 0.0
    latency_retrieval:   float = 0.0
    latency_generation:  float = 0.0
    latency_critic:      float = 0.0

    # Retrieval quality signals
    num_chunks:          int   = 0
    sections_retrieved:  list[str] = field(default_factory=list)   # which sections appeared
    papers_retrieved:    list[str] = field(default_factory=list)

    # User feedback (set later via /feedback endpoint)
    user_rating:    Optional[int] = None   # 1-5 thumbs from UI
    user_comment:   Optional[str] = None


def log_run(record: RunRecord) -> None:
    """Append a run record to the JSONL log file."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_runs(pipeline_type: Optional[str] = None) -> list[dict]:
    """Load all logged runs, optionally filtered by pipeline type."""
    if not LOG_PATH.exists():
        return []
    runs = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if pipeline_type is None or r.get("pipeline_type") == pipeline_type:
                        runs.append(r)
                except json.JSONDecodeError:
                    continue
    return runs


def update_feedback(run_id: str, rating: int, comment: str = "") -> bool:
    """
    Update user feedback for a specific run.
    
    Since JSONL is append-only, we rewrite the file with the updated record.
    Fine at this scale — only do this for user-triggered feedback.
    """
    if not LOG_PATH.exists():
        return False
    runs = load_runs()
    updated = False
    for r in runs:
        if r.get("run_id") == run_id:
            r["user_rating"]  = rating
            r["user_comment"] = comment
            updated = True
    if updated:
        with open(LOG_PATH, "w") as f:
            for r in runs:
                f.write(json.dumps(r) + "\n")
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT MANAGER — wraps a pipeline run and auto-logs timing
# ─────────────────────────────────────────────────────────────────────────────

class RunTimer:
    """
    Usage:
        timer = RunTimer()
        with timer.section("retrieval"):
            chunks = retrieve(...)
        with timer.section("generation"):
            answer = llm.invoke(...)
        record.latency_retrieval = timer["retrieval"]
        record.latency_total     = timer.total()
    """
    def __init__(self):
        self._sections: dict[str, float] = {}
        self._start = time.perf_counter()
        self._section_start: Optional[float] = None
        self._current: Optional[str] = None

    class _Section:
        def __init__(self, timer, name):
            self._t = timer
            self._n = name
        def __enter__(self):
            self._t._section_start = time.perf_counter()
            self._t._current       = self._n
        def __exit__(self, *_):
            elapsed = time.perf_counter() - self._t._section_start
            self._t._sections[self._n] = round(elapsed, 3)

    def section(self, name: str) -> "_Section":
        return self._Section(self, name)

    def __getitem__(self, name: str) -> float:
        return self._sections.get(name, 0.0)

    def total(self) -> float:
        return round(time.perf_counter() - self._start, 3)