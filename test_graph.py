import sys
sys.path.insert(0, "backend")

from dotenv import load_dotenv
load_dotenv()

from core.document_processor import load_index
from agents.graph import run_pipeline

store = load_index()

# Test 1
state = run_pipeline(
    "What is the main contribution of this paper?",
    store, k=5, max_retries=2
)

print("ANSWER:", state["answer"][:400])
print("SCORE:", state["critic_score"])
print("FLAGS:", state["hallucination_flags"])

# Test 2 (forces retry behavior)
state2 = run_pipeline(
    "What were the results?",
    store, k=5, max_retries=2
)

print("\nRETRIES NEEDED:", state2["retry_count"])