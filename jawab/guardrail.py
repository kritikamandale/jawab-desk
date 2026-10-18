"""Plain-code guardrail: language detection, deferrals, number and claim checking."""

import re
from decimal import Decimal, InvalidOperation

# --- language detection -----------------------------------------------------

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

HINGLISH_WORDS = [
    "kya", "hai", "kitna", "kitne", "kitni", "kab", "bhai", "didi", "mujhe",
    "nahi", "chahiye", "milega", "milegi", "kaise", "bhejo", "bhej", "batao",
    "karo", "kar", "karna", "theek", "thik", "accha", "acha", "paisa", "rupee",
    "rupees", "wala", "walon", "kaisa", "kaisi", "zaroor", "ji", "haan",
]
HINGLISH_RE = re.compile(r"\b(" + "|".join(HINGLISH_WORDS) + r")\b", re.IGNORECASE)


def detect_language(text: str) -> str:
    """Return "mr" (Devanagari), "hinglish" (Roman Hindi/Urdu words) or "en"."""
    if DEVANAGARI_RE.search(text):
        return "mr"
    if HINGLISH_RE.search(text):
        return "hinglish"
    return "en"


# --- fixed deferral templates ----------------------------------------------

DEFERRAL_TEMPLATES = {
    "en": "Thanks for checking! The seller will confirm this shortly and get back to you.",
    "hinglish": "Iska seller jaldi confirm karke aapko reply dega. Thoda wait karein!",
    "mr": "याबद्दल विक्रेता लवकर पडताळून कळवतील. थोडी वाट पहा!",
}

DEFERRAL_PHRASES = [
    "confirm",
    "get back to you",
    "check and get back",
    "will reply",
    "jaldi",          # hinglish "soon"
    "wait karein",
    "पडताळ",          # Marathi "verify/confirm"
    "कळव",            # Marathi "will let (you) know"
    "लवकर",
]

# --- numbers ----------------------------------------------------------------

_CURRENCY = r"(?:₹|rs\.?|inr)\s*"
_NUMBER_RE = re.compile(_CURRENCY + r"\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _normalize_number(token: str) -> str:
    token = re.sub(r"^[^\d]+", "", token).replace(",", "")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return token
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def extract_numbers(text: str) -> list:
    """Normalised numeric tokens: commas stripped, currency removed, decimals kept."""
    tokens = _NUMBER_RE.findall(text)
    cleaned = [t for t in _BARE_NUMBER_RE.findall(_NUMBER_RE.sub(" ", text))]
    seen, out = set(), []
    for token in tokens + cleaned:
        norm = _normalize_number(token)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# --- claim words ------------------------------------------------------------

CLAIM_WORDS = [
    "warranty", "guarantee", "refund", "return", "exchange",
    "free delivery", "cash on delivery", "cod", "discount", "negotiable",
]


def _claim_pattern(word: str) -> re.Pattern:
    if " " in word:
        return re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    return re.compile(r"\b" + re.escape(word) + r"(?:s|es|ies)?\b", re.IGNORECASE)


def sheet_text(item_sheet: dict) -> str:
    parts = []
    for key, value in item_sheet.items():
        if isinstance(value, list):
            parts.append(" ".join(str(v) for v in value))
        else:
            parts.append(f"{key} {value}")
    return " ".join(parts)


def contains_deferral_phrase(reply: str) -> bool:
    lowered = reply.lower()
    return any(phrase.lower() in lowered for phrase in DEFERRAL_PHRASES)


def is_supported(reply: str, item_sheet: dict, buyer_message: str):
    """Return (ok, reasons).

    Every number in the reply must appear in the item sheet or the buyer message.
    Claim words (warranty, refund, discount, ...) are blocked unless the same
    word appears in the item sheet.
    """
    reasons = []
    allowed_numbers = set(extract_numbers(sheet_text(item_sheet))) | set(
        extract_numbers(buyer_message)
    )
    for number in extract_numbers(reply):
        if number not in allowed_numbers:
            reasons.append(f"number {number} is not in the item sheet or the buyer message")
    sheet = sheet_text(item_sheet).lower()
    for word in CLAIM_WORDS:
        if _claim_pattern(word).search(reply) and word not in sheet:
            reasons.append(f"claim word '{word}' is not in the item sheet")
    return (len(reasons) == 0, reasons)
