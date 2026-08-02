import sys
sys.path.insert(0, "backend")

from dotenv import load_dotenv
load_dotenv()

from core import session_manager
from core.retriever import retrieve

PAPER_A = "data/papers/paper2.pdf"
PAPER_B = "data/papers/test_paper.pdf"

# ── Create two isolated sessions ─────────────────────────────────────────────
session_a = session_manager.create_session("Session A")
session_b = session_manager.create_session("Session B")

with open(PAPER_A, "rb") as f:
    session_manager.add_literature(session_a["id"], [("paper2.pdf", f.read())])

with open(PAPER_B, "rb") as f:
    session_manager.add_literature(session_b["id"], [("test_paper.pdf", f.read())])

print("Session A papers:", session_manager.get_indexed_literature(session_a["id"]))
print("Session B papers:", session_manager.get_indexed_literature(session_b["id"]))

# ── Isolation assertions ─────────────────────────────────────────────────────
store_a = session_manager.load_literature_store(session_a["id"])
store_b = session_manager.load_literature_store(session_b["id"])

chunks_a = retrieve("What is this paper about?", store_a, k=5, section_hint=False)
chunks_b = retrieve("What is this paper about?", store_b, k=5, section_hint=False)

titles_a = {c.metadata["paper_title"] for c in chunks_a}
titles_b = {c.metadata["paper_title"] for c in chunks_b}

print("\nRetrieved titles in Session A:", titles_a)
print("Retrieved titles in Session B:", titles_b)

assert titles_a, "Session A returned no chunks"
assert titles_b, "Session B returned no chunks"
assert titles_a.isdisjoint(titles_b), "LEAK: Session A and Session B share retrieved papers!"
assert "test_paper" not in titles_a, "LEAK: Session A retrieved Session B's paper"
assert "paper2" not in titles_b, "LEAK: Session B retrieved Session A's paper"

print("\n✅ Session isolation verified — no cross-session leakage.")
