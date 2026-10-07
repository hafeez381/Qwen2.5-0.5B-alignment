"""Run a colab CLI command with a wall-clock timeout and retries (guards the kernel-connect race hang).

The command runs in its own process group, and a timeout kills the whole group: killing only the direct child can
leave a grandchild holding the output pipes open, and then waiting for the output never returns.
"""
import os
import signal
import subprocess
import sys
import time

timeout, tries = int(sys.argv[1]), int(sys.argv[2])
cmd, stdin = sys.argv[3:], sys.stdin.read() if not sys.stdin.isatty() else None
for i in range(1, tries + 1):
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin is not None else None, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(stdin, timeout=timeout)
        print(f"[try {i}] rc={p.returncode}\n{out}", flush=True)
        sys.exit(p.returncode)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.stdout.close()
        p.wait()
        print(f"[try {i}] timed out after {timeout}s", flush=True)
        time.sleep(5)
sys.exit(124)
