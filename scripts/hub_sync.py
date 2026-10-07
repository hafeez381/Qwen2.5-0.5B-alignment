"""Move large artifacts through a private Hugging Face Hub repo and verify them by sha256.

push:  upload a local folder to <repo>/<path> and check every file's Hub LFS sha256 against the local file.
check: compare the Hub's sha256 for <repo>/<path> against local files (after `hf download`) or print them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

from alignment.io import sha256_file


def hub_hashes(api: HfApi, repo: str, path: str) -> dict[str, str]:
    files = [f for f in api.list_repo_files(repo) if f.startswith(path.rstrip("/") + "/")]
    out = {}
    for info in api.get_paths_info(repo, files, expand=True):
        out[info.path] = info.lfs.sha256 if info.lfs else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "check"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--path", required=True, help="folder inside the Hub repo")
    ap.add_argument("--local", help="local folder (push source, or downloaded copy to check)")
    args = ap.parse_args()
    api = HfApi()
    if args.action == "push":
        api.create_repo(args.repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=args.local, path_in_repo=args.path,
                          commit_message=f"upload {args.path}")
    hub = hub_hashes(api, args.repo, args.path)
    bad = 0
    for rel, h in sorted(hub.items()):
        local = Path(args.local) / Path(rel).relative_to(args.path) if args.local else None
        mine = sha256_file(local) if local and local.exists() else None
        ok = h is None or mine is None or mine == h
        bad += not ok
        print(f"{'OK' if ok else 'MISMATCH'} {rel} hub={h} local={mine}")
    print(("PASS" if not bad else "BREACH") + f": {len(hub)} file(s) under {args.repo}/{args.path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
