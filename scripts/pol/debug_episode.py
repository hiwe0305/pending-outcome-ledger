"""Run single expert episodes with full logs kept, and print why each was or was not kept (sim venv).

    python scripts/pol/debug_episode.py --task BatteriesSort-3 --seeds 1 5 11 --out data/pol_debug

Writes every episode (kept or not) like collect.py, plus one JSON line per seed with the expert_ok
diagnostics and a summary of the battery locations over time.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/pol")]


def location_timeline(logs, n):
    out, prev = [], None
    for t, r in enumerate(logs):
        loc = r.get("location", r.get("cube_location", []))[:n]
        if loc != prev:
            out.append((t, loc))
            prev = loc
    return out


def main():
    import collect
    from pol.config import load_config

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", default=str(ROOT / "data/pol_debug"))
    a = p.parse_args()
    cfg = load_config(ROOT / f"configs/pol/tasks/{a.task}.yaml")
    cfg["name"] = a.task
    for seed in a.seeds:
        r = collect.episode(cfg, seed, "train", a.out, keep_all=True)
        log = json.loads((Path(a.out) / a.task / f"{seed:06d}.log.json").read_text())
        n = (r["diag"] or {}).get("n", 6)
        r["timeline"] = location_timeline(log["logs"], n)[-12:]
        r["decisions"] = log["decisions"]
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
