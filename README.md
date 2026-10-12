# Small-Model Alignment: SFT and DPO on Qwen2.5-0.5B

Code for CS5326 (Advanced Generative AI) Programming Assignment 2. It post-trains `Qwen/Qwen2.5-0.5B` with supervised fine-tuning (SFT) and direct preference optimization (DPO), both written from scratch in PyTorch. DPO runs separately on HH-RLHF and UltraFeedback from the same SFT checkpoint. A third experiment tunes `HuggingFaceTB/SmolLM2-360M-Instruct` toward a chef persona with TRL `DPOTrainer` and a LoRA adapter. Each model is compared with its baseline through a blinded pairwise judge on Kaggle.

| Comparison | Judge rubric | Prompts | Win rate (ties = ½) |
|---|---|---|---|
| HH-RLHF DPO vs SFT | safety | 300 helpful/harmless | 0.605 [0.549, 0.659] |
| UltraFeedback DPO vs SFT | general | 300 AlpacaEval | 0.482 [0.425, 0.538] |
| LoRA-DPO vs SmolLM2 | persona style | 40 persona | 0.700 [0.546, 0.819] |
| LoRA-DPO vs SmolLM2 | usefulness and safety | 40 persona | 0.225 [0.123, 0.375] |

Judge `google/gemini-3.6-flash`, main pass, 95% Wilson intervals. Both A/B orders, retries and position bias are in `results/eval/*/summary_*.json`, and the analysis is in the report (`report/`).

## Setup

```bash
pip install -r requirements.txt
pip install -e . --no-deps
PYTHONPATH=. python -m pytest -q tests/
python scripts/check_boundary.py
```

The GPU runs used Python 3.12, torch 2.11.0+cu130, transformers 4.57.6, datasets 4.8.5, trl 0.27.2 and peft 0.17.1.

Every script takes `--mode {smoke,full}`. `smoke` swaps in tiny test models, runs a few steps on CPU and writes to `results/_smoke/` (never committed). `full` is the real run. All runs use seed 42.

## Compute

Training and generation ran on one Colab L4 (24 GB) through `scripts/colab_run.sh`, and judging ran on Kaggle. Checkpoints are pushed from the VM to a private Hugging Face Hub repository, so no weights are stored here.

```bash
bash scripts/colab_run.sh preflight                 # clean tree, tests, boundary check, ship git HEAD
bash scripts/colab_run.sh new L4
python3 scripts/colab_keepalive.py pa2-gpu 240 &    # stops a VM running only detached jobs from being reclaimed
bash scripts/colab_run.sh setup                     # install, rebuild and verify data, pull the SFT checkpoint
bash scripts/colab_run.sh launch NAME 'python -m ...'
bash scripts/colab_run.sh stop
```

## Reproducing the results

Run from the repository root in this order. Each step writes `results/<run>/` with per-item rows and a `summary.json` recomputed from them.

### Data
```bash
python scripts/prepare_data.py --mode full --verify     # rebuilds data/prepared/ and checks it against data/splits/
```

### Baseline generation and SFT
The base and SFT sanity generations predate the decoding freeze and used `repetition_penalty: 1.0` in `configs/generate.yaml`. Every later generation uses the committed value, 1.1.
```bash
python -m alignment.generate --mode full --model Qwen/Qwen2.5-0.5B --eval-set eval_sets/alpaca_eval_300_seed42.json --generator qwen2.5-0.5b-base --dataset alpaca_eval
python -m alignment.generate --mode full --model Qwen/Qwen2.5-0.5B --eval-set eval_sets/hh_rlhf_eval_300_seed42.json --generator qwen2.5-0.5b-base --dataset hh_rlhf_eval
python -m alignment.sft --mode full
python -m alignment.generate --mode full --model outputs/sft/best --eval-set eval_sets/alpaca_eval_300_seed42.json --generator sft --dataset alpaca_eval --limit 20
python -m alignment.generate --mode full --model outputs/sft/best --eval-set eval_sets/hh_rlhf_eval_300_seed42.json --generator sft --dataset hh_rlhf_eval --limit 5
python scripts/check_generations.py results/generations/sft__alpaca_eval__first20/generations.jsonl results/generations/sft__hh_rlhf_eval__first5/generations.jsonl
python scripts/plot_training.py results/sft/metrics.jsonl results/figures/sft_loss.png
```

### DPO (HH-RLHF and UltraFeedback)
```bash
python -m alignment.generate --mode full --model outputs/sft/best --eval-set eval_sets/hh_rlhf_eval_300_seed42.json --generator sft --dataset hh_rlhf_eval
python -m alignment.generate --mode full --model outputs/sft/best --eval-set eval_sets/alpaca_eval_300_seed42.json --generator sft --dataset alpaca_eval
python -m alignment.dpo --mode full --config configs/dpo_hh.yaml
python -m alignment.dpo --mode full --config configs/dpo_uf.yaml
python -m alignment.generate --mode full --model outputs/dpo_hh/best --tokenizer Qwen/Qwen2.5-0.5B --eval-set eval_sets/hh_rlhf_eval_300_seed42.json --generator dpo-hh --dataset hh_rlhf_eval
python -m alignment.generate --mode full --model outputs/dpo_uf/best --tokenizer Qwen/Qwen2.5-0.5B --eval-set eval_sets/alpaca_eval_300_seed42.json --generator dpo-uf --dataset alpaca_eval
python scripts/plot_dpo.py results/dpo_hh/metrics.jsonl results/figures/dpo_hh.png --title "HH-RLHF DPO"
python scripts/plot_dpo.py results/dpo_uf/metrics.jsonl results/figures/dpo_uf.png --title "UltraFeedback DPO"
```
Each DPO run stops before its first update unless step-0 validation shows accuracy 0 and loss ln 2 (policy equals reference). It also exits at step 150 if accuracy and margin both show no preference direction.

### LoRA persona DPO
```bash
python -m alignment.persona --mode full
python -m alignment.generate --mode full --model HuggingFaceTB/SmolLM2-360M-Instruct --eval-set eval_sets/persona_style_eval.json --generator smollm2-360m-instruct --dataset persona_style
python -m alignment.generate --mode full --model outputs/persona/merged --eval-set eval_sets/persona_style_eval.json --generator lora-persona --dataset persona_style
```

### Pairwise judging
```bash
G=results/generations
python scripts/build_pairs.py --focal $G/dpo-hh__hh_rlhf_eval/generations.jsonl --baseline $G/sft__hh_rlhf_eval/generations.jsonl --name hh_dpo_vs_sft
python scripts/build_pairs.py --focal $G/dpo-uf__alpaca_eval/generations.jsonl --baseline $G/sft__alpaca_eval/generations.jsonl --name uf_dpo_vs_sft
python scripts/build_pairs.py --focal $G/lora-persona__persona_style/generations.jsonl --baseline $G/smollm2-360m-instruct__persona_style/generations.jsonl --name lora_vs_base
```
Then create a Kaggle notebook at kaggle.com/benchmarks/tasks/new, import `notebooks/kaggle_judge.ipynb`, set `COMMIT` to the commit holding the pair files, and run it. It judges `hh_dpo_vs_sft` (safety rubric), `uf_dpo_vs_sft` (general) and `lora_vs_base` (style and usefulness) in both A/B orders with `scripts/judge_pairwise_kaggle.py`: 5 pairs per request, at least 5 s between attempts, up to 3 attempts per batch, unresolved pairs logged to a failure file, and finished pairs skipped on rerun. Put the downloaded `judged_*.jsonl` files into `results/eval/<name>/`, then:
```bash
python scripts/summarize_judgments.py --name hh_dpo_vs_sft --rubric safety
python scripts/summarize_judgments.py --name uf_dpo_vs_sft --rubric general
python scripts/summarize_judgments.py --name lora_vs_base --rubric style
python scripts/summarize_judgments.py --name lora_vs_base --rubric usefulness --compare style
python scripts/select_examples.py --name hh_dpo_vs_sft --rubric safety      # likewise for the other three
```

## Attribution

- **Course starter**: the `submission_adapter.py` signatures, `tests/test_public_adapter_smoke.py`, `tests/toy_lm.py`, `eval_sets/hh_rlhf_eval_300_seed42.json`, `data/persona_lora/` and `docs/assignment_manual.pdf`, with `requirements.txt` and `pyproject.toml` as starting points.
- **Libraries**: PyTorch, Transformers (model and tokenizer loading only in SFT/DPO), Datasets, TRL and PEFT (`alignment/persona.py` only), and `kaggle_benchmarks` (judging).
- **Models and data**: Qwen2.5-0.5B, SmolLM2-360M-Instruct, UltraChat-200k, HH-RLHF, UltraFeedback (binarized) and AlpacaEval, at the revisions pinned in `scripts/prepare_data.py`.
- No code was copied from public DPO implementations. Code was written with LLM assistance.

## References

- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.*
- Ouyang et al. (2022), *Training language models to follow instructions with human feedback.*
- Bai et al. (2022), *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.*
- Dubois et al. (2024), *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators.*
- Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models.*
