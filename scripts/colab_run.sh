#!/usr/bin/env bash
# GPU runs on Colab through the `colab` CLI. Run from the repo root. Every CLI call goes through
# scripts/colab_retry.py (wall-clock timeout + retries); long work runs detached on the VM.
#
#   bash scripts/colab_run.sh preflight              # clean tree, tests, boundary check, git-archive HEAD + GIT_SHA
#   bash scripts/colab_run.sh new [GPU]              # usage before/after, allocate (L4 default), save the browser URL
#   bash scripts/colab_run.sh setup                  # ship code + HF token, then scripts/vm_setup.sh detached (log: setup)
#   bash scripts/colab_run.sh launch NAME CMD...     # run CMD detached in the repo; log in outputs/logs/NAME.log
#   bash scripts/colab_run.sh status NAME            # last log lines, GPU memory/utilization, process alive?
#   bash scripts/colab_run.sh sh 'COMMAND'           # one short shell command on the VM (seconds, not minutes)
#   bash scripts/colab_run.sh fetch REMOTE LOCAL     # download a small file, verify sha256 against the VM's
#   bash scripts/colab_run.sh stop                   # release the VM, then show sessions and usage
#
# Env: SESSION (default pa2-gpu), GPU (default L4), HF_TOKEN_FILE (default ~/.cache/huggingface/token).
set -euo pipefail

SESSION="${SESSION:-pa2-gpu}"
REMOTE=/content/repo
SHIP=outputs/ship.tar.gz
RETRY="python3 scripts/colab_retry.py"

remote_py() {  # remote_py TIMEOUT <<'PY' ... PY
  local f; f="$(mktemp -d)/remote.py"; cat > "$f"
  $RETRY "$1" 3 colab exec -s "$SESSION" --timeout "$1" -f "$f"
}

remote_sh() {  # remote_sh TIMEOUT 'shell command'   (base64 so no quoting can break the Python wrapper)
  local b64; b64="$(printf '%s' "$2" | base64 | tr -d '\n')"
  remote_py "$1" <<PY
import base64, subprocess
r = subprocess.run(base64.b64decode("$b64").decode(), shell=True, capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:]); print("exit", r.returncode)
PY
}

cmd="${1:-}"; shift || true
case "$cmd" in
preflight)
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "commit first: the VM runs HEAD" >&2; exit 1; }
  PYTHONPATH=. ~/miniconda3/envs/pa2-gai/bin/python -m pytest -q tests/
  ~/miniconda3/envs/pa2-gai/bin/python scripts/check_boundary.py
  mkdir -p outputs && git rev-parse HEAD > outputs/GIT_SHA
  git archive --format=tar HEAD > outputs/ship.tar
  tar -rf outputs/ship.tar -C outputs GIT_SHA && gzip -f outputs/ship.tar
  echo "HEAD $(cat outputs/GIT_SHA)  tarball sha256 $(shasum -a 256 "$SHIP" | cut -d' ' -f1)  size $(du -h "$SHIP" | cut -f1)"
  ;;

new)
  gpu="${1:-${GPU:-L4}}"
  colab usage
  colab sessions
  $RETRY 300 1 colab new -s "$SESSION" --gpu "$gpu"
  colab url -s "$SESSION" | tee outputs/colab_url.txt
  colab usage
  ;;

setup)
  [[ -f "$SHIP" ]] || { echo "run preflight first" >&2; exit 1; }
  $RETRY 180 3 colab upload -s "$SESSION" "$SHIP" /content/ship.tar.gz
  tokfile="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
  [[ -f "$tokfile" ]] || { echo "no HF token at $tokfile: run 'hf auth login' first (snapshots go to the Hub)" >&2; exit 1; }
  $RETRY 60 3 colab upload -s "$SESSION" "$tokfile" /content/hf_token
  remote_sh 60 "mkdir -p /root/.cache/huggingface $REMOTE/outputs/logs && mv /content/hf_token /root/.cache/huggingface/token && \
    tar -xzf /content/ship.tar.gz -C $REMOTE && cat $REMOTE/GIT_SHA"
  "$0" launch setup bash scripts/vm_setup.sh
  ;;

launch)
  name="${1:?NAME required}"; shift
  b64="$(printf '%s' "$*" | base64 | tr -d '\n')"
  remote_py 60 <<PY
import base64, shlex, subprocess
cmd = base64.b64decode("$b64").decode()
print("launching:", cmd)
subprocess.Popen(f"cd $REMOTE && TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup bash -c {shlex.quote(cmd)} > outputs/logs/$name.log 2>&1; echo EXIT=\$? >> outputs/logs/$name.log",
                 shell=True, start_new_session=True)
PY
  ;;

status)
  name="${1:?NAME required}"
  remote_sh 60 "tail -n 8 $REMOTE/outputs/logs/$name.log; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; pgrep -af 'python' | grep -v -E 'colab|jupyter|ipykernel' || echo 'no python job running'"
  ;;

sh)
  remote_sh 120 "cd $REMOTE && $1"
  ;;

fetch)
  src="${1:?REMOTE path (relative to the repo) required}"; dst="${2:?LOCAL path required}"
  mkdir -p "$(dirname "$dst")"
  remote_sh 60 "cd $REMOTE && sha256sum $src"
  $RETRY 230 3 colab download -s "$SESSION" "$REMOTE/$src" "$dst"
  echo "local: $(shasum -a 256 "$dst" | cut -d' ' -f1)  (must equal the remote line above)"
  ;;

stop)
  $RETRY 120 3 colab stop -s "$SESSION" || true
  colab sessions
  colab usage
  ;;

*)
  sed -n '2,15p' "$0"; exit 1
  ;;
esac
