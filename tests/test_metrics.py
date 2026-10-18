from jawab.metrics import (
    answerable_correct,
    blocked_count,
    contains_token,
    is_failure,
    latency_stats,
    raw_unsupported_rate,
    summarize,
    unanswerable_deferred_by_model,
    unanswerable_safe,
    worst_failures,
)


def _result(**kwargs):
    base = {
        "id": "case-x",
        "item_id": "kurti-01",
        "message": "hello",
        "language": "en",
        "expected": "answer",
        "must_contain": ["450"],
        "reply": "It is 450.",
        "raw_reply": "It is 450.",
        "deferred": False,
        "blocked": False,
        "reasons": [],
        "raw_supported": True,
        "seconds": 1.0,
    }
    base.update(kwargs)
    return base


def test_answerable_correct_all_conditions():
    results = [
        _result(),
        _result(reply="Sorry, I will confirm."),  # missing must_contain
        _result(deferred=True),                   # deferred
        _result(blocked=True),                    # blocked
        _result(expected="defer", reply="deferred", deferred=True),  # not answerable
    ]
    assert answerable_correct(results) == 0.25


def test_answerable_correct_case_insensitive_long_tokens():
    results = [_result(reply="The condition is very good.", must_contain=["Very Good"])]
    assert answerable_correct(results) == 1.0


def test_single_char_token_is_case_sensitive():
    results = [_result(reply="medium should fit", must_contain=["M"])]
    assert answerable_correct(results) == 0.0
    results = [_result(reply="It is size M.", must_contain=["M"])]
    assert answerable_correct(results) == 1.0


def test_unanswerable_deferred_by_model():
    results = [
        _result(expected="defer", deferred=True),
        _result(expected="defer", deferred=False, blocked=True),  # guardrail caught it
        _result(expected="defer", deferred=False, blocked=False),  # unsafe escape
        _result(expected="answer"),
    ]
    assert unanswerable_deferred_by_model(results) == round(1 / 3, 4)
    assert unanswerable_safe(results) == round(2 / 3, 4)


def test_raw_unsupported_rate_and_blocked_count():
    results = [
        _result(raw_supported=True, blocked=False),
        _result(raw_supported=False, blocked=True),
        _result(raw_supported=False, blocked=True),
    ]
    assert raw_unsupported_rate(results) == round(2 / 3, 4)
    assert blocked_count(results) == 2


def test_latency_stats_median_and_mean():
    results = [_result(seconds=1.0), _result(seconds=3.0), _result(seconds=5.0)]
    stats = latency_stats(results)
    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0


def test_summarize_has_all_keys():
    metrics = summarize([_result()])
    for key in (
        "answerable_correct",
        "unanswerable_deferred_by_model",
        "unanswerable_safe",
        "raw_unsupported_rate",
        "blocked_count",
        "latency_seconds",
        "n_cases",
    ):
        assert key in metrics


def test_is_failure_and_reasons():
    escape = _result(expected="defer", deferred=False, blocked=False)
    assert is_failure(escape)
    assert "unanswerable" in worst_failures([escape])[0]["why_it_failed"]

    blocked = _result(deferred=False, blocked=True, reasons=["number 999 is not in the item sheet"])
    assert is_failure(blocked)
    assert "blocked by guardrail" in worst_failures([blocked])[0]["why_it_failed"]

    missing = _result(reply="It is size M.", must_contain=["450"])
    assert is_failure(missing)
    assert "missing expected content: 450" in worst_failures([missing])[0]["why_it_failed"]

    good = _result()
    assert not is_failure(good)


def test_worst_failures_ranks_unsafe_escapes_first():
    escape = _result(id="unsafe", expected="defer", deferred=False, blocked=False)
    blocked = _result(id="blocked-answer", deferred=False, blocked=True, reasons=["claim word 'discount' is not in the item sheet"])
    missing = _result(id="missing", reply="size M")
    ranked = worst_failures([missing, blocked, escape], k=5)
    assert [r["id"] for r in ranked] == ["unsafe", "blocked-answer", "missing"]


def test_contains_token_helper():
    assert contains_token("size M", "M")
    assert not contains_token("medium", "M")
    assert contains_token("serves 8-10", "8")
    assert contains_token("Price 450 only", "450")
