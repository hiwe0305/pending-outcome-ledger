"""Bounded public-source audit downloader. No credentials, weights or full dataset."""

import hashlib, json, pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS = {
    f"BatteriesChecker{level}-{n}-VLA-v0": f"batteries_checker_{level.lower()}_{n}_vla_v0"
    for level in ("Easy", "Hard")
    for n in (3, 6)
}


def fetch(url, path, limit):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "jem-bounded-audit"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"File exceeds budget: {path}")
    path.write_bytes(data)


def main():
    (ROOT / "runs").mkdir(exist_ok=True)
    rev = json.loads((ROOT / "configs/manifest.json").read_text())["dataset"][
        "revision"
    ]
    base = f"https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot/resolve/{rev}"
    for task, folder in TASKS.items():
        paths = [
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
            "source_rlds_metadata.json",
            "data/chunk-000/file-000.parquet",
        ]
        for rel in paths:
            dest = ROOT / "data/audit" / folder / rel
            fetch(
                f"{base}/{folder}/{rel}",
                dest,
                50_000_000,
            )
            print(task, rel, dest.stat().st_size, flush=True)
    fetch(f"{base}/README.md", ROOT / "data/audit/README.md", 100_000)
    files = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (ROOT / "data/audit").rglob("*")
        if f.is_file()
    }
    (ROOT / "runs/download-hashes.json").write_text(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
