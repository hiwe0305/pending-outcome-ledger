"""Write each task's step budget = P95(kept expert episode length) x 1.15 into its task yaml.

    python scripts/pol/calibrate_budget.py --data data/pol --tasks BatteriesSort-3 SealedBin-5 ...

Run on the pilot collection (>= 200 kept episodes per task), before Stage B; the budget then stays
fixed (it enters the provenance hash through the task config).
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def budget(steps, factor=1.15):
    return int(np.ceil(np.percentile(steps, 95) * factor))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data/pol"))
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--min-episodes", type=int, default=200)
    a = p.parse_args()
    for task in a.tasks:
        steps = [e["steps"] for f in glob.glob(f"{a.data}/{task}/summary-train-*.json")
                 for e in json.load(open(f))["episodes"] if e["kept"]]
        if len(steps) < a.min_episodes:
            print(f"{task}: only {len(steps)} kept episodes (< {a.min_episodes}); not written")
            continue
        b = budget(steps)
        y = ROOT / f"configs/pol/tasks/{task}.yaml"
        text = y.read_text()
        line = f"max_steps: {b}          # P95 x 1.15 of {len(steps)} expert episodes (calibrate_budget.py)"
        text = re.sub(r"^max_steps:.*$", line, text, flags=re.M) if re.search(r"^max_steps:", text, re.M) else text + line + "\n"
        y.write_text(text)
        print(f"{task}: max_steps = {b}")


if __name__ == "__main__":
    main()
