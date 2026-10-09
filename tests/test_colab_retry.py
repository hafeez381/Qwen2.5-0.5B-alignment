import subprocess
import sys
import time


def test_timeout_fires_even_when_stdin_is_an_open_pipe():
    holder = subprocess.Popen(["sleep", "30"], stdout=subprocess.PIPE)
    t0 = time.time()
    r = subprocess.run([sys.executable, "scripts/colab_retry.py", "1", "1", "sleep", "30"], stdin=holder.stdout,
                       capture_output=True, text=True, timeout=20)
    holder.kill()
    assert r.returncode == 124 and time.time() - t0 < 10
