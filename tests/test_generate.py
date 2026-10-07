import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.format import render_prompt
from alignment.generate import generate_batch, generation_config

CFG = {"max_new_tokens": 12, "repetition_penalty": 1.0}


def test_decoding_is_greedy_and_ignores_the_model_sampling_defaults():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5").eval()
    assert model.generation_config.do_sample  # the tiny model ships sampling defaults
    gen = generation_config(CFG, tok)
    assert gen["eos_token_id"] == [151643, 151645]
    prompts = [render_prompt(tok, "Say hi."), render_prompt(tok, "Write a long sentence about the sea.")]
    torch.manual_seed(0)
    a = generate_batch(model, tok, prompts, gen)
    torch.manual_seed(1)
    b = generate_batch(model, tok, prompts, gen)
    assert a == b
    assert model.generation_config.do_sample is False and model.generation_config.repetition_penalty == 1.0
    single = generate_batch(model, tok, prompts[:1], gen)
    assert single[0] == a[0]
