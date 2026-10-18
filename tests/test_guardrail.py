from jawab.guardrail import (
    DEFERRAL_TEMPLATES,
    contains_deferral_phrase,
    detect_language,
    extract_numbers,
    is_supported,
)

SHEET = {
    "id": "kurti-01",
    "title": "Thrifted cotton kurti, mustard yellow, size M",
    "price_inr": 450,
    "size_or_measurements": "M (chest 38 inches, length 44 inches)",
    "flaws": ["small 1 cm mark near the hem"],
    "delivery": "Ships in 2-3 days, delivery charge 60, free delivery above 1000",
    "payment": "GPay before dispatch, no returns",
}


def test_extract_plain_number():
    assert extract_numbers("The price is 450") == ["450"]


def test_extract_comma_and_currency():
    assert extract_numbers("Rs. 1,200 total") == ["1200"]
    assert extract_numbers("₹999") == ["999"]


def test_extract_decimal():
    assert extract_numbers("5.5 cm long") == ["5.5"]
    assert extract_numbers("₹99.50 only") == ["99.5"]


def test_extract_range_splits():
    assert extract_numbers("ships in 2-3 days") == ["2", "3"]


def test_extract_deduplicates_and_keeps_order():
    assert extract_numbers("450 for M, chest 38 for M") == ["450", "38"]


def test_number_in_sheet_is_supported():
    ok, reasons = is_supported("It is 450 for size M.", SHEET, "kitne ki hai?")
    assert ok
    assert reasons == []


def test_number_from_buyer_message_is_allowed():
    ok, _ = is_supported("We cannot do 400, sorry.", SHEET, "400 mein de do")
    assert ok


def test_unsupported_number_is_blocked():
    ok, reasons = is_supported("It is 499 only.", SHEET, "price kya hai?")
    assert not ok
    assert any("499" in r for r in reasons)


def test_claim_word_not_in_sheet_is_blocked():
    ok, reasons = is_supported("No discount on this one.", SHEET, "discount do na")
    assert not ok
    assert any("discount" in r for r in reasons)


def test_claim_word_in_sheet_is_allowed():
    sheet = dict(SHEET, notes="No returns. free delivery above 1000.")
    ok, reasons = is_supported("Sorry, no returns on this.", sheet, "return kar sakte ho?")
    assert ok, reasons


def test_cod_is_matched_as_word_not_substring():
    # "second" contains "cod" but must not trigger the COD claim check.
    ok, reasons = is_supported("I will ship it second.", SHEET, "kab milega?")
    assert ok, reasons
    ok, reasons = is_supported("COD is available.", SHEET, "COD ho sakta hai?")
    assert not ok


def test_plural_claim_forms_are_blocked():
    ok, reasons = is_supported("We offer exchanges.", SHEET, "exchange chahiye")
    assert not ok
    assert any("exchange" in r for r in reasons)


def test_language_detection_english():
    assert detect_language("What is the chest measurement?") == "en"


def test_language_detection_hinglish():
    assert detect_language("bhai ye kurti ka price kya hai?") == "hinglish"
    assert detect_language("kab tak milega delivery?") == "hinglish"


def test_language_detection_marathi():
    assert detect_language("तुमचा मोबाईल नंबर काय आहे?") == "mr"
    assert detect_language("केक अंड्यांशिवाय मिळेल का?") == "mr"


def test_english_words_do_not_trigger_hinglish():
    assert detect_language("Do you have size M in stock?") == "en"


def test_deferral_templates_exist_for_all_languages():
    assert set(DEFERRAL_TEMPLATES) == {"en", "hinglish", "mr"}
    for text in DEFERRAL_TEMPLATES.values():
        assert text.strip()


def test_contains_deferral_phrase():
    assert contains_deferral_phrase("The seller will confirm shortly.")
    assert contains_deferral_phrase("Seller jaldi confirm karke reply dega.")
    assert not contains_deferral_phrase("It is 450 and size M.")
