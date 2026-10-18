"""One reply from the terminal.

Usage:
    python scripts/demo_cli.py "ye kurti kitne ki hai?"
    python scripts/demo_cli.py --item cake-01 "Do you deliver?"
    JAWAB_MOCK=1 python scripts/demo_cli.py "price kya hai?"
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jawab.model import DEFAULT_MODEL, HFModel, MockModel  # noqa: E402
from jawab.reply import generate_reply  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Draft one buyer reply with Jawab Desk")
    parser.add_argument("message", help="the buyer's DM")
    parser.add_argument("--item", default="kurti-01", help="item sheet id (default: kurti-01)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    args = parser.parse_args()

    with open(ROOT / "data" / "item_sheets.json", encoding="utf-8") as fh:
        sheets = {item["id"]: item for item in json.load(fh)["items"]}
    if args.item not in sheets:
        parser.error(f"unknown item id {args.item!r}; choose from {', '.join(sheets)}")
    item = sheets[args.item]

    if __import__("os").environ.get("JAWAB_MOCK") == "1":
        model = MockModel()
        used = "MockModel (JAWAB_MOCK=1)"
    else:
        model = HFModel(args.model)
        used = args.model

    result = generate_reply(item, args.message, model)

    print(f"item      : {item['title']}")
    print(f"buyer DM  : {args.message}")
    print(f"model     : {used}")
    print(f"language  : {result['language']}")
    print(f"status    : {'deferred' if result['deferred'] else 'blocked' if result['blocked'] else 'answered'}")
    print(f"seconds   : {result['seconds']}")
    print(f"reply     : {result['reply']}")
    if result["reasons"]:
        print(f"reasons   : {'; '.join(result['reasons'])}")
    print(f"raw reply : {result['raw_reply']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
