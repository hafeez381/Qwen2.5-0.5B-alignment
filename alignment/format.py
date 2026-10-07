from __future__ import annotations


def render_prompt(tokenizer, instruction: str) -> str:
    """The one prompt format used by SFT, DPO and generation: the tokenizer's own chat template, single user turn."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )
