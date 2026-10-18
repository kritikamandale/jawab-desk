from jawab.guardrail import DEFERRAL_TEMPLATES
from jawab.model import MockModel
from jawab.prompt import SYSTEM_PROMPT, build_user_prompt
from jawab.reply import generate_reply

SHEET = {
    "id": "kurti-01",
    "title": "Thrifted cotton kurti, mustard yellow, size M",
    "price_inr": 450,
    "size_or_measurements": "M (chest 38 inches, length 44 inches)",
    "flaws": ["small 1 cm mark near the hem"],
    "delivery": "Ships in 2-3 days, delivery charge 60",
    "payment": "GPay before dispatch",
    "notes": "No returns.",
}


def test_confirm_is_deferred():
    result = generate_reply(SHEET, "Do you ship internationally?", MockModel(response="[CONFIRM]"))
    assert result["deferred"] is True
    assert result["blocked"] is False
    assert result["reply"] == DEFERRAL_TEMPLATES["en"]
    assert result["raw_reply"] == "[CONFIRM]"


def test_deferral_language_follows_buyer_language():
    hinglish = generate_reply(SHEET, "kab tak milega?", MockModel(response="[CONFIRM]"))
    assert hinglish["language"] == "hinglish"
    assert hinglish["reply"] == DEFERRAL_TEMPLATES["hinglish"]

    marathi = generate_reply(SHEET, "तुमचा मोबाईल नंबर काय आहे?", MockModel(response="[CONFIRM]"))
    assert marathi["language"] == "mr"
    assert marathi["reply"] == DEFERRAL_TEMPLATES["mr"]


def test_deferral_phrase_alone_defers():
    result = generate_reply(SHEET, "price kya hai?", MockModel(response="The seller will confirm shortly."))
    assert result["deferred"] is True
    assert result["reply"] == DEFERRAL_TEMPLATES["hinglish"]


def test_supported_reply_passes_through():
    result = generate_reply(SHEET, "ye kurti kitne ki hai?", MockModel(response="It is 450 for size M."))
    assert result["reply"] == "It is 450 for size M."
    assert result["deferred"] is False
    assert result["blocked"] is False
    assert result["reasons"] == []


def test_unsupported_number_blocks_and_defers_final_reply():
    result = generate_reply(SHEET, "price kya hai?", MockModel(response="It is 499 only."))
    assert result["blocked"] is True
    assert result["deferred"] is False
    assert result["reply"] == DEFERRAL_TEMPLATES["hinglish"]
    assert result["raw_reply"] == "It is 499 only."
    assert any("499" in r for r in result["reasons"])


def test_blocked_claims_use_buyer_language_template():
    result = generate_reply(SHEET, "return kar sakte ho kya?", MockModel(response="Don't worry, it has a warranty."))
    assert result["blocked"] is True
    assert result["reply"] == DEFERRAL_TEMPLATES["hinglish"]


def test_result_shape_and_timing():
    result = generate_reply(SHEET, "hello", MockModel(response="Hi, size M."))
    for key in ("reply", "language", "deferred", "blocked", "raw_reply", "reasons", "seconds"):
        assert key in result
    assert result["seconds"] >= 0
    assert result["language"] == "en"


def test_prompt_contains_sheet_facts_and_buyer_message():
    prompt = build_user_prompt(SHEET, "price kya hai?")
    assert "price_inr: 450" in prompt
    assert "Buyer message: price kya hai?" in prompt
    assert "synthetic" not in prompt
    assert "[CONFIRM]" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT


def test_default_mock_quotes_only_sheet_values():
    model = MockModel()
    system = SYSTEM_PROMPT
    user = build_user_prompt(SHEET, "anything at all?")
    reply = model.generate(system, user)
    assert "450" in reply or "Mustard" in reply.lower() or "mustard" in reply.lower()
