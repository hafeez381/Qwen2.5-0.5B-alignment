"""Run a colab CLI command with a wall-clock timeout and retries (guards the kernel-connect race hang)."""
import subprocess, sys, time
timeout, tries = int(sys.argv[1]), int(sys.argv[2])
cmd, stdin = sys.argv[3:], sys.stdin.read() if not sys.stdin.isatty() else None
for i in range(1, tries + 1):
    try:
        r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
        print(f"[try {i}] rc={r.returncode}\n{r.stdout}{r.stderr}", flush=True)
        sys.exit(r.returncode)
    except subprocess.TimeoutExpired:
        print(f"[try {i}] timed out after {timeout}s", flush=True); time.sleep(5)
sys.exit(124)
