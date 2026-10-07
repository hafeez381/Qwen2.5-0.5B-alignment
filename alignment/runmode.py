"""`--mode smoke|full`: smoke swaps each model for a tiny one with the same tokenizer, so every path runs on a laptop CPU."""
from __future__ import annotations

SMOKE_MODELS = {
    "Qwen/Qwen2.5-0.5B": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
}


def add_mode_arg(parser) -> None:
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)


def resolve_model(model_id: str, mode: str) -> str:
    if mode not in ("smoke", "full"):
        raise ValueError(mode)
    return SMOKE_MODELS[model_id] if mode == "smoke" else model_id
