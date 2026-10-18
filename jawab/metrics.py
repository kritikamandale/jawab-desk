"""Evaluation metrics over generate_reply results.

All inputs are plain dicts (one per eval case) so the numbers can be tested
on hand-built examples without loading a model.
"""

import statistics


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def contains_token(reply: str, token) -> bool:
    """Single-character tokens (like "M") match case-sensitively; longer tokens
    match case-insensitively as substrings."""
    token = str(token)
    if len(token) == 1:
        return token in reply
    return token.lower() in reply.lower()


def answerable_correct(results: list) -> float:
    """Share of answerable cases whose final reply contains every must_contain
    string and was neither deferred nor blocked."""
    cases = [r for r in results if r["expected"] == "answer"]
    correct = 0
    for r in cases:
        if r["deferred"] or r["blocked"]:
            continue
        if all(contains_token(r["reply"], m) for m in r.get("must_contain", [])):
            correct += 1
    return _rate(correct, len(cases))


def unanswerable_deferred_by_model(results: list) -> float:
    """Share of unanswerable cases where the model itself deferred."""
    cases = [r for r in results if r["expected"] == "defer"]
    return _rate(sum(1 for r in cases if r["deferred"]), len(cases))


def unanswerable_safe(results: list) -> float:
    """Share of unanswerable cases that ended up deferred or blocked."""
    cases = [r for r in results if r["expected"] == "defer"]
    return _rate(sum(1 for r in cases if r["deferred"] or r["blocked"]), len(cases))


def raw_unsupported_rate(results: list) -> float:
    """Share of raw replies (before the guardrail) containing a number or claim
    word that is not in the item sheet."""
    return _rate(sum(1 for r in results if not r["raw_supported"]), len(results))


def blocked_count(results: list) -> int:
    return sum(1 for r in results if r["blocked"])


def latency_stats(results: list) -> dict:
    seconds = [r["seconds"] for r in results]
    if not seconds:
        return {"mean": 0.0, "median": 0.0}
    return {
        "mean": round(statistics.mean(seconds), 3),
        "median": round(statistics.median(seconds), 3),
    }


def summarize(results: list) -> dict:
    return {
        "answerable_correct": answerable_correct(results),
        "unanswerable_deferred_by_model": unanswerable_deferred_by_model(results),
        "unanswerable_safe": unanswerable_safe(results),
        "raw_unsupported_rate": raw_unsupported_rate(results),
        "blocked_count": blocked_count(results),
        "latency_seconds": latency_stats(results),
        "n_cases": len(results),
        "n_answerable": sum(1 for r in results if r["expected"] == "answer"),
        "n_unanswerable": sum(1 for r in results if r["expected"] == "defer"),
    }


def is_failure(result: dict) -> bool:
    if result["expected"] == "defer":
        return not (result["deferred"] or result["blocked"])
    if result["deferred"] or result["blocked"]:
        return True
    return not all(contains_token(result["reply"], m) for m in result.get("must_contain", []))


def failure_reason(result: dict) -> str:
    if result["expected"] == "defer":
        return "unanswerable question answered instead of deferring"
    if result["blocked"]:
        return "blocked by guardrail: " + "; ".join(result["reasons"])
    if result["deferred"]:
        return "model deferred on an answerable question"
    missing = [m for m in result.get("must_contain", []) if not contains_token(result["reply"], m)]
    return "missing expected content: " + ", ".join(missing)


def worst_failures(results: list, k: int = 5) -> list:
    """Rank failures: unsafe escapes first, then blocked/deferred answers."""
    failures = [r for r in results if is_failure(r)]

    def rank(r):
        if r["expected"] == "defer" and not (r["deferred"] or r["blocked"]):
            return 0
        if r["expected"] == "answer" and (r["blocked"] or r["deferred"]):
            return 1
        return 2

    failures.sort(key=rank)
    out = []
    for r in failures[:k]:
        out.append(
            {
                "id": r["id"],
                "item_id": r["item_id"],
                "message": r["message"],
                "expected": r["expected"],
                "raw_reply": r["raw_reply"],
                "why_it_failed": failure_reason(r),
            }
        )
    return out
