"""Gradio demo for Jawab Desk.

Run:  python app.py
Mock: JAWAB_MOCK=1 python app.py   (no model download needed)
"""

import json
import os
from pathlib import Path

import gradio as gr

from jawab.model import DEFAULT_MODEL, HFModel, MockModel
from jawab.reply import generate_reply

ROOT = Path(__file__).resolve().parent
MOCK = os.environ.get("JAWAB_MOCK") == "1"

MODELS = [
    DEFAULT_MODEL,
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

SAMPLE_SHEET = {
    "id": "kurti-01",
    "title": "Thrifted cotton kurti, mustard yellow, size M",
    "price_inr": 450,
    "size_or_measurements": "M (chest 38 inches, length 44 inches)",
    "condition": "Very good, worn once",
    "flaws": ["small 1 cm mark near the hem, hidden when worn"],
    "delivery": "Ships in 2-3 days from Pune, delivery charge 60, free delivery above 1000",
    "payment": "GPay or bank transfer before dispatch, Cash on Delivery available",
    "notes": "No returns or exchanges on thrifted items. Dry clean only.",
}
SAMPLE_SHEET_JSON = json.dumps(SAMPLE_SHEET, indent=2, ensure_ascii=False)
SAMPLE_MESSAGE = "bhai ye kurti ka price kya hai?"

_cache = {}


def _get_model(model_name: str):
    """Load the model lazily, once per model name."""
    if MOCK:
        return MockModel()
    if model_name not in _cache:
        _cache[model_name] = HFModel(model_name)
    return _cache[model_name]


def draft(sheet_json: str, message: str, model_name: str):
    if not message or not message.strip():
        raise gr.Error("Please paste the buyer's message first.")
    try:
        sheet = json.loads(sheet_json)
        if not isinstance(sheet, dict):
            raise ValueError("item sheet must be a JSON object")
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Item sheet is not valid JSON: {exc}")

    result = generate_reply(sheet, message.strip(), _get_model(model_name))
    if result["deferred"]:
        status = "deferred - the model did not confirm from the sheet, so the seller will confirm"
    elif result["blocked"]:
        status = "blocked - the plain-code guardrail stopped an unsupported number or claim"
    else:
        status = "answered - checked against the item sheet"
    details = {
        "language detected": result["language"],
        "model": "MockModel (JAWAB_MOCK=1)" if MOCK else model_name,
        "seconds": result["seconds"],
        "guardrail reasons": result["reasons"] or "none",
    }
    return result["reply"], status, result["raw_reply"], json.dumps(details, indent=2, ensure_ascii=False)


with gr.Blocks(title="Jawab Desk") as demo:
    gr.Markdown(
        "# Jawab Desk\n"
        "Fill the item sheet once. Paste the buyer's DM. Get a reply the model is only allowed "
        "to ground in the sheet - anything else becomes a deferral, and a plain-code guardrail "
        "double-checks every number and policy claim.\n\n"
        + ("**Running in mock mode (JAWAB_MOCK=1) - no model is downloaded.**" if MOCK else "")
    )
    with gr.Row():
        with gr.Column():
            sheet_box = gr.Textbox(
                label="Item sheet (JSON)",
                value=SAMPLE_SHEET_JSON,
                lines=14,
            )
            msg_box = gr.Textbox(label="Buyer message", value=SAMPLE_MESSAGE, lines=2)
            model_dd = gr.Dropdown(
                choices=MODELS,
                value=MODELS[0],
                label="Model",
                info="Loaded lazily on the first reply. CPU only.",
            )
            btn = gr.Button("Draft reply", variant="primary")
        with gr.Column():
            reply_box = gr.Textbox(label="Final reply (what the buyer sees)", lines=4)
            status_box = gr.Textbox(label="Status", lines=2)
            with gr.Accordion("Raw model reply + details", open=False):
                raw_box = gr.Textbox(label="Raw reply (before guardrail)", lines=4)
                detail_box = gr.Textbox(label="Details", lines=6)

    btn.click(draft, inputs=[sheet_box, msg_box, model_dd], outputs=[reply_box, status_box, raw_box, detail_box])

if __name__ == "__main__":
    demo.launch()
