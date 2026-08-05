"""
metrics.py
----------
Aggregates run logs into dashboard-ready metrics.

METRICS WE TRACK AND WHY:

  hallucination_rate:
    % of runs with at least one flag. This is the headline metric —
    the whole point of the Critic Agent is to drive this toward 0.
    If it's high, your retrieval or answering prompt needs work.

  avg_critic_score:
    Mean quality score across all runs. Should trend upward as you
    tune chunk size, k, and prompts.

  retry_rate:
    % of runs that needed ≥1 retry. High retry rate = your first-pass
    retrieval is often insufficient. Consider increasing k or improving
    section detection.

  pass_rate:
    % of runs that got verdict=PASS. Direct measure of system reliability.

  avg_latency:
    Mean end-to-end time. Important for UX — if >15s, users notice.
    Breakdown by stage tells you WHERE the bottleneck is.

  score_over_time:
    Trending critic scores — are you improving the system?

  section_distribution:
    Which paper sections are being retrieved most often.
    Useful for diagnosing retrieval bias.
"""

from collections import defaultdict, Counter
from datetime    import datetime, timezone
from typing      import Optional
from evaluation.logger import load_runs

# p95/p99 are noise below this many samples — a single slow run can swing
# them wildly, so the dashboard shows "not enough data" instead of a number
# that looks precise but isn't.
MIN_RUNS_FOR_TAIL_LATENCY = 20


def _percentile(sorted_values: list[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile over an already-sorted list — no
    numpy dependency, matching this module's existing dependency-light style."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def compute_metrics(pipeline_type: str = "qa", session_id: Optional[str] = None) -> dict:
    runs = load_runs(pipeline_type, session_id)
    if not runs:
        return {"total_runs": 0}

    n = len(runs)

    # ── Core quality metrics ───────────────────────────────────────────────
    scores       = [r["critic_score"]  for r in runs if r.get("critic_score")]
    retries      = [r["retry_count"]   for r in runs]
    latencies    = [r["latency_total"] for r in runs if r.get("latency_total")]
    flags_counts = [len(r.get("hallucination_flags", [])) for r in runs]
    verdicts     = Counter(r.get("verdict", "UNKNOWN") for r in runs)

    hallucination_rate = sum(1 for f in flags_counts if f > 0) / n
    retry_rate         = sum(1 for r in retries if r > 0) / n
    pass_rate          = verdicts.get("PASS", 0) / n

    # ── Latency breakdown ──────────────────────────────────────────────────
    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0

    sorted_latencies = sorted(latencies)
    enough_for_tail   = len(sorted_latencies) >= MIN_RUNS_FOR_TAIL_LATENCY
    p50 = _percentile(sorted_latencies, 50)
    p95 = _percentile(sorted_latencies, 95) if enough_for_tail else None
    p99 = _percentile(sorted_latencies, 99) if enough_for_tail else None

    latency_breakdown = {
        "p50":             round(p50, 3) if p50 is not None else 0,
        "p95":             round(p95, 3) if p95 is not None else None,
        "p99":             round(p99, 3) if p99 is not None else None,
        "enough_for_tail": enough_for_tail,
        "retrieval":       avg([r.get("latency_retrieval", 0) for r in runs]),
        "generation":      avg([r.get("latency_generation", 0) for r in runs]),
        "critic":          avg([r.get("latency_critic", 0) for r in runs]),
    }

    # ── Token / cost tracking ───────────────────────────────────────────────
    # total_tokens is None (not 0) for runs logged before this field existed
    # — those are excluded from cost/token aggregates entirely rather than
    # silently counted as zero-cost.
    token_tracked = [r for r in runs if r.get("total_tokens") is not None]
    total_tokens  = sum(r["total_tokens"] for r in token_tracked)
    total_cost    = sum(r.get("estimated_cost_usd") or 0.0 for r in token_tracked)

    token_stats = {
        "tracked_runs":       len(token_tracked),
        "untracked_runs":     n - len(token_tracked),
        "total_tokens":       total_tokens,
        "total_cost_usd":     round(total_cost, 4),
        "avg_tokens_per_run": round(total_tokens / len(token_tracked), 1) if token_tracked else 0,
        "any_estimated":      any(r.get("tokens_estimated") for r in token_tracked),
    }

    # ── Trends (last 20 runs, chronological) ──────────────────────────────
    recent = sorted(runs, key=lambda r: r.get("timestamp",""))[-20:]
    score_trend = [
        {"timestamp": r["timestamp"][:10], "score": r.get("critic_score", 0)}
        for r in recent
    ]
    latency_trend = [
        {"timestamp": r["timestamp"][:10], "latency": r.get("latency_total", 0)}
        for r in recent
    ]

    # ── Section distribution ───────────────────────────────────────────────
    section_counts = Counter()
    for r in runs:
        for s in r.get("sections_retrieved", []):
            section_counts[s] += 1

    # ── Failure analysis ───────────────────────────────────────────────────
    # Most common hallucination flags (for debugging)
    all_flags = []
    for r in runs:
        all_flags.extend(r.get("hallucination_flags", []))
    top_flags = Counter(all_flags).most_common(5)

    # Worst-performing queries (score < 6)
    weak_runs = [
        {"query": r["query"][:80], "score": r["critic_score"],
         "verdict": r.get("verdict",""), "retries": r["retry_count"]}
        for r in runs if r.get("critic_score", 10) < 6
    ][:5]

    # ── User feedback stats ────────────────────────────────────────────────
    rated = [r for r in runs if r.get("user_rating")]
    user_stats = {
        "rated_count":  len(rated),
        "avg_rating":   avg([r["user_rating"] for r in rated]),
    } if rated else {"rated_count": 0, "avg_rating": 0}

    return {
        "total_runs":          n,
        "hallucination_rate":  round(hallucination_rate * 100, 1),   # %
        "retry_rate":          round(retry_rate * 100, 1),            # %
        "pass_rate":           round(pass_rate * 100, 1),             # %
        "avg_critic_score":    round(avg(scores), 2),
        "avg_flags_per_run":   round(avg(flags_counts), 2),
        "verdict_counts":      dict(verdicts),
        "latency":             latency_breakdown,
        "score_trend":         score_trend,
        "latency_trend":       latency_trend,
        "section_distribution":dict(section_counts.most_common(8)),
        "top_flags":           top_flags,
        "weak_runs":           weak_runs,
        "user_feedback":       user_stats,
        "tokens":              token_stats,
    }


def compute_all_metrics(session_id: Optional[str] = None) -> dict:
    """Compute metrics across all pipeline types, optionally scoped to one session."""
    return {
        "qa":         compute_metrics("qa", session_id),
        "comparison": compute_metrics("comparison", session_id),
        "ideas":      compute_metrics("ideas", session_id),
        "critique":   compute_metrics("critique", session_id),
        "combined":   compute_metrics(None, session_id),
    }