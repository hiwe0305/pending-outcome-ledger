"""Fixed task subset, official seed stream, paired seed-cluster bootstrap.

Pairing is enforced on provenance too: within a training seed, every compared
method must share the fields in SHARED_PROVENANCE (benchmark, initialization,
labels, data and budgets); only architecture-specific fields may differ.
"""

import hashlib, json
import numpy as np
from .data import SORT_TASKS, TASKS

SHARED_PROVENANCE = ("benchmark_revision", "init_checkpoint", "label_hash", "train_ids_hash",
                     "chunk_size", "tuning_budget", "code_hash")

START_SEED = 4242424242
EPISODES = 50


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def check_provenance(records, methods):
    """Reject comparisons whose runs differ in anything but the architecture."""
    seen = {}
    for r in records:
        if r["variant"] not in methods:
            continue
        prov = r.get("provenance")
        if not isinstance(prov, dict) or any(k not in prov for k in SHARED_PROVENANCE):
            raise ValueError("Record lacks run provenance")
        key = r["training_seed"]
        shared = tuple(prov[k] for k in SHARED_PROVENANCE)
        if seen.setdefault(key, shared) != shared:
            raise ValueError(f"Provenance differs between compared runs at training seed {key}")


def paired_report(
    records,
    reference="KEM",
    candidate="EEM",
    tasks=None,
    repeats=2000,
    seed=17029,
    margin=0.03,
    canonical=True,
):
    tasks = sorted(tasks or SORT_TASKS[:2])
    valid_tasks = set(TASKS) | set(SORT_TASKS)
    methods = (reference, candidate)
    seeds = sorted({r["training_seed"] for r in records})
    if not seeds or repeats < 1 or margin <= 0:
        raise ValueError("Invalid evaluation design")
    data = {m: {} for m in methods}
    for r in records:
        if r["variant"] not in data:
            continue
        if (
            r.get("kind") != "model_evaluation"
            or r["task"] not in valid_tasks
            or not isinstance(r["success_once"], bool)
        ):
            raise ValueError("Not an official model evaluation record")
        if r["task"] not in tasks:
            continue
        key = (r["training_seed"], r["episode_seed"], r["task"])
        if key in data[r["variant"]]:
            raise ValueError("Duplicate result")
        data[r["variant"]][key] = float(r["success_once"])
    check_provenance(records, methods)
    if set(data[reference]) != set(data[candidate]) or not data[reference]:
        raise ValueError("Paired evaluation cells differ or empty")
    episode_seeds = sorted({k[1] for k in data[reference]})
    if canonical and episode_seeds != list(range(START_SEED, START_SEED + EPISODES)):
        raise ValueError("Official 50-episode stream required")
    expected = {(s, e, t) for s in seeds for e in episode_seeds for t in tasks}
    if set(data[reference]) != expected:
        raise ValueError("Missing custom-subset cells")
    a = np.array(
        [
            [[data[reference][s, e, t] for t in tasks] for e in episode_seeds]
            for s in seeds
        ]
    )
    b = np.array(
        [
            [[data[candidate][s, e, t] for t in tasks] for e in episode_seeds]
            for s in seeds
        ]
    )
    rng = np.random.default_rng(seed)
    samples = []
    absolute = {m: [] for m in methods}
    for _ in range(repeats):
        si = rng.integers(len(seeds), size=len(seeds))
        ei = rng.integers(len(episode_seeds), size=len(episode_seeds))
        aa = a[si][:, ei]
        bb = b[si][:, ei]
        samples.append((bb - aa).mean())
        absolute[reference].append(aa.mean())
        absolute[candidate].append(bb.mean())
    ci = np.quantile(samples, [0.025, 0.975]).tolist()
    conclusion = "uncertain"
    if ci[0] > margin:
        conclusion = "supports_practical_candidate_gain_on_fixed_subset"
    elif ci[0] >= -margin and ci[1] <= margin:
        conclusion = "practically_equivalent"
    elif ci[1] < -margin:
        conclusion = "candidate_practically_worse"
    return {
        "reporting": "custom_subset",
        "reference": reference,
        "candidate": candidate,
        "per_task": {
            m: dict(zip(tasks, x.mean((0, 1)).tolist()))
            for m, x in [(reference, a), (candidate, b)]
        },
        "macro": {reference: float(a.mean()), candidate: float(b.mean())},
        "macro_ci95": {
            m: np.quantile(absolute[m], [0.025, 0.975]).tolist() for m in methods
        },
        "difference": float((b - a).mean()),
        "difference_ci95": ci,
        "practical_equivalence_interval": [-margin, margin],
        "conclusion": conclusion,
        "ci_method": "paired bootstrap training-seed and shared episode-seed clusters; fixed tasks",
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
        "training_seeds": seeds,
        "evaluation_seeds": episode_seeds,
        "limitations": (
            ["single training seed cannot estimate training-seed variation"]
            if len(seeds) == 1
            else []
        ),
    }
