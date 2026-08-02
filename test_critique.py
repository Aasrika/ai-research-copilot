"""
test_critique.py
-----------------
Standalone CLI smoke test for the Mode 2 critique pipeline, mirroring
the existing test_graph.py / test_sessions.py convention.

TODO (before Phase 4 / UI): test_paper.pdf is used here as a stand-in
"draft" purely to exercise the pipeline end-to-end without exceptions.
It was indexed alongside paper2.pdf as literature, so there may be
accidental content overlap between "draft" and "literature" — this is
NOT a real validation of classification quality. Before building the
Critique UI page, replace this with a handcrafted draft containing
known-SUPPORTED and known-CONTRADICTED claims against the literature,
so classification accuracy can actually be checked.
"""

import sys
sys.path.insert(0, "backend")

from dotenv import load_dotenv
load_dotenv()

from core import session_manager
from agents.critique_graph import run_critique

session = session_manager.create_session("Critique Test Session")

with open("data/papers/paper2.pdf", "rb") as f:
    session_manager.add_literature(session["id"], [("paper2.pdf", f.read())])

with open("data/papers/test_paper.pdf", "rb") as f:
    draft_path = session_manager.set_draft(session["id"], "test_paper.pdf", f.read())

literature_store = session_manager.load_literature_store(session["id"])

report = run_critique(
    session_id=session["id"],
    draft_path=str(draft_path),
    literature_store=literature_store,
    max_claims=8,
)

print(report.model_dump_json(indent=2))

assert len(report.claims) >= 1, "Expected at least one claim to be classified"
print(f"\n✅ Critique pipeline ran end-to-end — {len(report.claims)} claims classified.")
