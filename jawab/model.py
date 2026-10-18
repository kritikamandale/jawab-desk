"""Model loading and generation.

Hugging Face `transformers` only. The real model is loaded once and cached.
`MockModel` has the same interface so tests and the demo can run without a download.
"""

import os

DEFAULT_MODEL = os.environ.get("JAWAB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

_tokenizer = None
_model = None
_model_name = None


class MockModel:
    """Drop-in stand-in for the real model. Same generate(system, user) interface."""

    def __init__(self, response=None):
        self.response = response

    def generate(self, system: str, user: str) -> str:
        if self.response is not None:
            return self.response
        # Deterministic demo behaviour: quote the first item detail line.
        # Never invents anything, because everything it says comes from the prompt.
        for line in user.splitlines():
            if ":" in line and not line.startswith("Buyer message"):
                key, value = line.split(":", 1)
                if value.strip() and key.strip() not in ("Item details", "Buyer message"):
                    return f"As per the item details, {key.strip()}: {value.strip()}."
        return "[CONFIRM]"


def load_model(model_name: str = None):
    """Load (or reuse) the tokenizer and causal LM for `model_name`."""
    global _tokenizer, _model, _model_name
    model_name = model_name or DEFAULT_MODEL
    if _model is not None and _model_name == model_name:
        return _tokenizer, _model
    if _model is not None:
        unload_model()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", **kwargs)
    model.eval()
    _tokenizer, _model, _model_name = tokenizer, model, model_name
    return _tokenizer, _model


def unload_model():
    """Free the currently loaded model so another one can be loaded."""
    global _tokenizer, _model, _model_name
    if _model is not None:
        import gc

        _model = None
        _tokenizer = None
        _model_name = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def generate(model_name: str, system: str, user: str, max_new_tokens: int = 100) -> str:
    """Greedy generation with the chat template of `model_name`."""
    tokenizer, model = load_model(model_name)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    import torch

    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class HFModel:
    """Thin wrapper giving the real model the same interface as MockModel."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or DEFAULT_MODEL

    def generate(self, system: str, user: str) -> str:
        return generate(self.model_name, system, user)
