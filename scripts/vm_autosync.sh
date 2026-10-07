#!/usr/bin/env bash
# Runs detached on the Colab VM: push each new snapshot of the watched runs to the Hub, and once DONE_LOG has an
# EXIT line, push best/final checkpoints and a tarball of the result files. Nothing waits on a laptop-side fetch.
# usage: bash scripts/vm_autosync.sh TAG DONE_LOG "run1 run2 ..." "extra_dir1 extra_dir2 ..." "result_path1 ..."
set -u
REPO=hafeez381/cs5326-pa2-checkpoints
tag="$1"; done_log="$2"; runs="$3"; extras="$4"; results="$5"
declare -A last
while true; do
  for r in $runs; do
    s=$(head -c 40 "outputs/$r/snapshot/state.json" 2>/dev/null)
    if [ -n "$s" ] && [ "$s" != "${last[$r]:-}" ]; then
      rm -rf "outputs/$r/snapshot_push" && cp -r "outputs/$r/snapshot" "outputs/$r/snapshot_push" && echo "PUSH $r $s"
      python scripts/hub_sync.py push --repo $REPO --path "$r/snapshot" --local "outputs/$r/snapshot_push" | grep -E "PASS|BREACH|MISMATCH"
      last[$r]="$s"
    fi
  done
  grep -q "^EXIT=" "$done_log" 2>/dev/null && break
  sleep 60
done
echo "CHAIN_DONE $(grep '^EXIT=' "$done_log")"
mkdir -p "outputs/results_$tag"
tar -czf "outputs/results_$tag/results_$tag.tgz" $results outputs/logs 2>/dev/null
sha256sum "outputs/results_$tag/results_$tag.tgz"
python scripts/hub_sync.py push --repo $REPO --path "results/$tag" --local "outputs/results_$tag" | grep -E "PASS|BREACH|MISMATCH"
for d in $extras; do
  [ -d "outputs/$d" ] && python scripts/hub_sync.py push --repo $REPO --path "$d" --local "outputs/$d" | grep -E "PASS|BREACH|MISMATCH"
done
echo AUTOSYNC_DONE
