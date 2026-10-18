"""Reply orchestration: generate, then defer or block via the guardrail."""

import time

from .guardrail import (
    DEFERRAL_TEMPLATES,
    contains_deferral_phrase,
    detect_language,
    is_supported,
)
from .prompt import SYSTEM_PROMPT, build_user_prompt


def generate_reply(item_sheet: dict, buyer_message: str, model) -> dict:
    language = detect_language(buyer_message)
    started = time.perf_counter()
    raw_reply = str(model.generate(SYSTEM_PROMPT, build_user_prompt(item_sheet, buyer_message))).strip()
    elapsed = time.perf_counter() - started

    reasons = []
    deferred = "[CONFIRM]" in raw_reply or contains_deferral_phrase(raw_reply)
    if deferred:
        reasons.append("model deferred ([CONFIRM] or deferral phrase in raw reply)")

    blocked = False
    if not deferred:
        ok, reasons = is_supported(raw_reply, item_sheet, buyer_message)
        if not ok:
            blocked = True

    if deferred or blocked:
        final_reply = DEFERRAL_TEMPLATES[language]
    else:
        final_reply = raw_reply

    return {
        "reply": final_reply,
        "language": language,
        "deferred": deferred,
        "blocked": blocked,
        "raw_reply": raw_reply,
        "reasons": reasons,
        "seconds": round(elapsed, 3),
    }
