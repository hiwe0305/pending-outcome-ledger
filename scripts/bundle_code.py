"""Pack the code for the GPU server (excludes data, runs, checkpoints, external, .venv)."""
import hashlib, json, pathlib, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARTS = ["src", "configs", "scripts", "tests", "docs", "third_party", "README.md", "pyproject.toml", ".gitignore"]

if __name__ == "__main__":
    out = ROOT / "runs/eem-code.zip"
    out.parent.mkdir(exist_ok=True)
    files = []
    for name in PARTS:
        p = ROOT / name
        files += [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, pathlib.Path("joint-execution-memory") / f.relative_to(ROOT))
    print(json.dumps({"archive": str(out.relative_to(ROOT)), "files": len(files), "bytes": out.stat().st_size,
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))
