"""Explicit opt-in on another machine: download exact initialization (~7B weights).
Never run automatically; no rented resources, no alternate initialization.
"""

import json, pathlib
from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    manifest = json.loads((ROOT / "configs/manifest.json").read_text())
    ckpt = manifest["checkpoint"]
    path = snapshot_download(
        repo_id=ckpt["id"],
        revision=ckpt["revision"],
        cache_dir=str(ROOT / "checkpoints/hf"),
    )
    print(path)
