"""System prompt and prompt builder for Jawab Desk."""

SYSTEM_PROMPT = """\
You help a small online seller reply to a buyer's direct message.
Use ONLY the item details given below.
Reply in the same language and script the buyer used: English, Hinglish written in Roman letters, or Marathi in Devanagari.
Keep the reply to 1 to 3 short, friendly sentences.
If the answer is not in the item details, reply with exactly [CONFIRM] and nothing else.
Never invent prices, sizes, materials, delivery times, discounts or policies.\
"""

_SKIP_KEYS = {"id", "synthetic"}


def format_item_sheet(item_sheet: dict) -> str:
    lines = []
    for key, value in item_sheet.items():
        if key in _SKIP_KEYS:
            continue
        if isinstance(value, list):
            value = "; ".join(str(v) for v in value) if value else "none"
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_user_prompt(item_sheet: dict, buyer_message: str) -> str:
    return (
        "Item details:\n"
        f"{format_item_sheet(item_sheet)}\n"
        "\n"
        f"Buyer message: {buyer_message}"
    )


def build_messages(item_sheet: dict, buyer_message: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item_sheet, buyer_message)},
    ]
