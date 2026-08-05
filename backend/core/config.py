"""
config.py
---------
Single source of truth for all settings.

WHY THIS MATTERS:
Hardcoding values like chunk_size=800 in 5 different files means
changing it requires hunting through your whole codebase. This file
fixes that. It also makes your project look professional on GitHub.
"""

from pathlib import Path


BASE_DIR      = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = BASE_DIR / "data"
PAPERS_DIR    = DATA_DIR / "papers"
INDICES_DIR   = DATA_DIR / "indices"
SESSIONS_DIR  = DATA_DIR / "sessions"


EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # local HuggingFace sentence-transformer, free, no API key
EMBEDDING_DIM     = 384


ANSWERING_MODEL = "llama-3.3-70b-versatile"
CRITIC_MODEL = "llama-3.3-70b-versatile"
IDEA_MODEL = "llama-3.1-8b-instant"
COMPARISON_MODEL = "llama-3.1-8b-instant"
CRITIQUE_MODEL = "llama-3.3-70b-versatile"  # larger model for claim extraction/classification
TEMPERATURE = 0


# ─────────────────────────────────────────────────────────────────────────────
# COST ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
# Groq itself is free/low-cost for this project's usage — these rates
# approximate what the equivalent OPENAI production cost would be, purely so
# the evaluation dashboard can demonstrate cost-awareness. $ per million
# tokens, keyed by the ACTUAL model name used for a given call (not by
# pipeline/agent), so the estimate stays correct automatically if a model
# assignment above changes later (e.g. downgrading ANSWERING_MODEL back to
# 8B while keeping CRITIC_MODEL on 70B).
MODEL_COST_PER_MILLION_TOKENS = {
    "llama-3.1-8b-instant":    {"input": 0.15, "output": 0.60},    # GPT-4o-mini equivalent
    "llama-3.3-70b-versatile": {"input": 2.50, "output": 10.00},   # GPT-4o equivalent
}
DEFAULT_COST_PER_MILLION_TOKENS = {"input": 0.15, "output": 0.60}


CHUNK_SIZE        = 800    
CHUNK_OVERLAP     = 150    


TOP_K             = 5     
MMR_LAMBDA        = 0.7    
                           

SECTION_KEYWORDS  = {
    "abstract":     ["abstract"],
    "introduction": ["introduction", "background", "motivation"],
    "methods":      ["method", "methodology", "approach", "architecture",
                     "model", "framework", "algorithm", "experiment setup"],
    "results":      ["result", "finding", "performance", "accuracy",
                     "evaluation", "benchmark", "ablation"],
    "discussion":   ["discussion", "analysis", "limitation", "future work",
                     "conclusion", "we show", "we find"],
    "related_work": ["related work", "prior work", "literature", "survey"],
}