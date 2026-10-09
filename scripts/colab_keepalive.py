"""Keep a Colab VM from being reclaimed as idle while detached jobs run.

Colab reclaims a VM roughly 30 min after the last client-side execution, even if a background process is using the
GPU. This sends a trivial `colab exec` every INTERVAL seconds, each guarded by a watchdog that kills the whole process
group, until outputs/keepalive.stop exists.
usage: python3 scripts/colab_keepalive.py SESSION [INTERVAL]
"""
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime

session = sys.argv[1]
interval = int(sys.argv[2]) if len(sys.argv) > 2 else 240
probe = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
probe.write("print('alive')\n")
probe.close()
misses = 0
while not os.path.exists("outputs/keepalive.stop"):
    p = subprocess.Popen(["colab", "exec", "-s", session, "--timeout", "30", "-f", probe.name], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=60)
        ok = "alive" in out
        misses = 0 if ok else misses + 1
        status = "ok" if ok else out.strip()[-120:]
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        misses += 1
        status = "timeout"
    print(f"{datetime.now():%H:%M:%S} {status}", flush=True)
    if "not found" in status:
        break
    time.sleep(interval if misses == 0 else 20)
