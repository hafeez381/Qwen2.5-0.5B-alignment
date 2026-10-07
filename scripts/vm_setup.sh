#!/usr/bin/env bash
# Runs detached on the Colab VM from the repo root: install deps without touching torch, record the environment,
# rebuild the frozen splits and check them against the committed ids and hashes.
set -euo pipefail
pip -q uninstall -y torchao || true
pip -q install -r requirements.txt
pip -q install -e . --no-deps
{ nvidia-smi; python -c 'import torch, transformers, datasets, trl, peft; print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "datasets", datasets.__version__, "trl", trl.__version__, "peft", peft.__version__)'; cat GIT_SHA; } | tee outputs/env.txt
python scripts/prepare_data.py --mode full --verify
