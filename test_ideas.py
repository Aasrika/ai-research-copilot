import sys
sys.path.insert(0, "backend")

from dotenv import load_dotenv
load_dotenv()

from core.document_processor import load_index, get_indexed_papers
from agents.idea_agent import generate_research_ideas

# Load vector DB
store = load_index()

# Get available papers
papers = get_indexed_papers()
print("Indexed papers:", papers)

if not papers:
    print("❌ No papers indexed. Upload first.")
    exit()

paper = papers[0]   # pick first paper

ideas = generate_research_ideas(
    vector_store = store,
    paper_filter = paper,
    focus_area   = ""   # you can try: "methodology", "dataset bias", etc.
)

print("\n📌 SUMMARY:")
print(ideas["summary"])

print(f"\nExplicit limitations: {len(ideas['explicit_limitations'])}")
print(f"Implicit limitations: {len(ideas['implicit_limitations'])}")
print(f"Open questions:       {len(ideas['open_questions'])}")
print(f"Experiment ideas:     {len(ideas['experiment_ideas'])}")

# Print one example idea
if ideas["experiment_ideas"]:
    e = ideas["experiment_ideas"][0]
    print(f"\n💡 First Idea: {e['title']}")
    print(f"   {e['description']}")
    print(f"   Impact: {e['expected_impact']}")
    print(f"   Difficulty: {e['difficulty']}")