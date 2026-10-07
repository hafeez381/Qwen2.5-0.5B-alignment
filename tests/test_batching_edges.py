import pytest
import torch

from alignment.batching import IGNORE, build_dpo_batch, build_lm_batch, encode_pair
from tests.helpers import NoEosTokenizer, NoPadTokenizer, PadIsEosTokenizer
from tests.toy_lm import ToyTokenizer

TOK = ToyTokenizer()
EOS = TOK.eos_token_id


def row_parts(batch, row):
    labels, mask = batch["labels"][row], batch["attention_mask"][row]
    return labels[labels.ne(IGNORE)].tolist(), int(mask.sum())


def test_eos_is_appended_before_truncation():
    ids, n_prompt = encode_pair(TOK, "Q: ", "abc", max_length=16)
    assert ids[-1] == EOS and n_prompt == 3 and len(ids) == 7
    ids, _ = encode_pair(TOK, "Q: ", "abc", max_length=6)
    assert EOS not in ids and len(ids) == 6
    ids, _ = encode_pair(TOK, "Q: ", "abc", max_length=7)
    assert ids[-1] == EOS


def test_prompt_filling_or_exceeding_max_length_leaves_no_response_labels():
    for max_length in (3, 2):
        batch = build_lm_batch(TOK, ["Q: "], ["abc"], max_length)
        assert batch["labels"].eq(IGNORE).all()
        assert batch["attention_mask"].sum() == max_length
    batch = build_lm_batch(TOK, ["Q: "], ["abc"], max_length=4)
    assert row_parts(batch, 0)[0] == TOK.encode("a")


def test_pad_equal_to_eos_keeps_the_eos_label():
    tok = PadIsEosTokenizer()
    batch = build_lm_batch(tok, ["Q: ", "Task: "], ["ok", "yes"], max_length=16)
    for row, resp in enumerate(["ok", "yes"]):
        labels, n_real = row_parts(batch, row)
        assert labels == tok.encode(resp) + [tok.eos_token_id]
        assert batch["labels"][row, n_real:].eq(IGNORE).all()
        assert batch["attention_mask"][row, n_real:].eq(0).all()


def test_tokenizer_without_pad_or_eos():
    batch = build_lm_batch(NoPadTokenizer(), ["Q: "], ["ok"], max_length=8)
    assert batch["input_ids"][0, 6:].eq(TOK.eos_token_id).all()
    assert row_parts(batch, 0)[0] == TOK.encode("ok") + [EOS]
    batch = build_lm_batch(NoEosTokenizer(), ["Q: "], ["ok"], max_length=8)
    assert row_parts(batch, 0)[0] == TOK.encode("ok")


def test_shapes_and_padding_width():
    batch = build_lm_batch(TOK, ["Q: ", "Longer task: "], ["a", "bcd"], max_length=32)
    assert {k: tuple(v.shape) for k, v in batch.items()} == {k: (2, 32) for k in batch}
    longest = build_lm_batch(TOK, ["Q: ", "Longer task: "], ["a", "bcd"], max_length=32, pad_to="longest")
    assert longest["input_ids"].shape == (2, len("Longer task: ") + 4)
    for k in batch:
        assert torch.equal(batch[k][:, : longest[k].shape[1]], longest[k])


def test_row_order_is_preserved():
    prompts, responses = ["A: ", "Bb: ", "Ccc: "], ["x", "yy", "zzz"]
    batch = build_lm_batch(TOK, prompts, responses, max_length=16)
    perm = [2, 0, 1]
    permuted = build_lm_batch(TOK, [prompts[i] for i in perm], [responses[i] for i in perm], max_length=16)
    for k in batch:
        assert torch.equal(batch[k][perm], permuted[k])


def test_dpo_batch_formats_the_prompt_identically_on_both_sides():
    batch = build_dpo_batch(TOK, ["Q: ", "Task: "], ["good", "yes"], ["bad", "no"], max_length=16)
    for row, prompt in enumerate(["Q: ", "Task: "]):
        n = len(TOK.encode(prompt))
        assert torch.equal(batch["chosen"]["input_ids"][row, :n], batch["rejected"]["input_ids"][row, :n])
        assert batch["chosen"]["labels"][row, :n].eq(IGNORE).all()


def test_real_qwen_tokenizer_pad_is_eos():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    assert tok.pad_token_id == tok.eos_token_id == 151643
    batch = build_lm_batch(tok, ["<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"], ["Hello!"], max_length=64)
    labels = batch["labels"][0][batch["labels"][0].ne(IGNORE)].tolist()
    assert labels == tok.encode("Hello!", add_special_tokens=False) + [151643]
