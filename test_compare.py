import sys
sys.path.insert(0, "backend")

from dotenv import load_dotenv
load_dotenv()

from core.document_processor import load_index, get_indexed_papers
from agents.comparison_agent import run_comparison

store  = load_index()
papers = get_indexed_papers()

print("Indexed papers:", papers)

if len(papers) < 2:
    print("❌ Need at least 2 papers indexed")
    exit()

result = run_comparison(
    paper_a      = papers[0],
    paper_b      = papers[1],
    aspect       = "Methodology",   # try others later
    vector_store = store,
)

structured = result["structured"]

print("\n⚖️ VERDICT:")
print(structured.get("verdict", ""))

print("\n🧠 SYNTHESIS:")
print(structured.get("synthesis", ""))