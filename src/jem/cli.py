"""Command line entry points. Heavyweight paths fail closed at unresolved gates.

    python -m jem.cli preflight
    python -m jem.cli synthetic-smoke
    python -m jem.cli real-data-smoke
    python -m jem.cli train --variant-config configs/variants/EEM.json --data data/batteriessort/<task> ...
    python -m jem.cli report runs/eval/records.json --reference KEM --candidate EEM --output report.json
"""

import argparse, hashlib, importlib.util, json, platform, random, resource, shutil, subprocess, time
from pathlib import Path
import numpy as np
import torch
from .data import Episode, TrainNormalizer, calls, episode_split
from .evaluation import config_hash, paired_report
from .fixtures import TinyBackbone, synthetic_episode
from .policy import VARIANTS, JointPolicy
from .training import parameter_counts, train_episode

ROOT = Path(__file__).resolve().parents[2]
SMOKE = [("B0", None)] + [(v, g) for v in VARIANTS[1:] for g in (None, "il")]


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def sha256_files(paths):
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def code_hash():
    return sha256_files((ROOT / "src/jem").rglob("*.py"))


def preflight(out):
    modules = ["torch", "numpy", "pytest", "transformers", "peft", "mani_skill", "sapien", "mplib",
               "mikasa_robo_suite", "timm", "av", "pyarrow"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dependencies": {m: bool(importlib.util.find_spec(m)) for m in modules},
        "gpus": gpu.stdout.strip().splitlines() if gpu and gpu.returncode == 0 else [],
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 2**30, 1),
        "code_hash": code_hash(),
    }
    write(out, report)
    print(json.dumps(report, indent=2))


def synthetic_smoke(out):
    torch.set_num_threads(2)
    logs = {}
    episode, labels = synthetic_episode()
    for variant, gate in SMOKE:
        torch.manual_seed(17)
        policy = JointPolicy(TinyBackbone(), variant, gate=gate)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        start = time.perf_counter()
        history = train_episode(policy, episode, optimizer, labels if variant != "B0" else None, synthetic=True)
        policy.reset()
        pred = policy.inference(episode.observation(0))
        logs[f"{variant}/{gate}"] = {"parameters": parameter_counts(policy), "final_losses": history[-1]["losses"],
                                     "seconds": time.perf_counter() - start, "action_finite": bool(np.isfinite(pred).all())}
    write(out, {"kind": "synthetic_implementation_smoke", "benchmark_performance": False,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "variants": logs})
    print("Synthetic train/backward/inference passed:", ", ".join(logs))


def real_data_smoke(out):
    torch.set_num_threads(2)
    reports = []
    for path in sorted((ROOT / "data/samples").glob("*.npz")):
        ep = Episode.load(path)
        results = {}
        for variant, gate in SMOKE:
            torch.manual_seed(17)
            policy = JointPolicy(TinyBackbone(), variant, gate=gate).eval()
            policy.reset()
            with torch.no_grad():
                for i, call in enumerate(calls(ep)):
                    pred, _ = policy.step(call.observation, call.executed, call.dt, call.frames)
                    if not torch.isfinite(pred).all():
                        raise RuntimeError("Invalid real-input smoke")
                    if i == 2:
                        break
            results[f"{variant}/{gate}"] = {"calls": i + 1, "action_shape": list(pred.shape)}
        reports.append({"episode_id": ep.episode_id, "frames": len(ep.actions), "variants": results})
    if not reports:
        raise RuntimeError("No sample episodes in data/samples (run scripts/extract_real_samples.py)")
    write(out, {"kind": "real_input_tiny_fixture_inference", "oft_7b": False, "episodes": reports})
    print(f"Real-input tiny-fixture forward passed on {len(reports)} episodes; no OFT model result")


def load_dataset(folders):
    episodes, labels = {}, {}
    for folder in folders:
        for npz in sorted(Path(folder).glob("*.npz")):
            ep = Episode.load(npz)
            episodes[ep.episode_id] = ep
            lab = npz.with_suffix(".labels.json")
            if lab.exists():
                labels[ep.episode_id] = {int(k): v for k, v in json.loads(lab.read_text()).items()}
    return episodes, labels


def train(args):
    from .backbone import configure_lora, load_local_oft
    from .checkpointing import save_adapter
    from .data import pilot_normalizers
    from .supervision import verify_certificate

    manifest = json.loads((ROOT / "configs/manifest.json").read_text())
    config = json.loads((ROOT / "configs/common.json").read_text())
    vcfg = json.loads(Path(args.variant_config).read_text())
    required = ["G1_data", "G2_env", "backbone_injection", "resource_profile"] + (["G3_labels"] if vcfg["variant"] != "B0" else [])
    pending = [g for g in required if manifest["gates"].get(g) != "verified"]
    if pending:
        raise RuntimeError("Training gates unresolved: " + ", ".join(pending))
    episodes, labels = load_dataset(args.data)
    if not episodes:
        raise ValueError("No episodes found")
    certificate = None
    if vcfg["variant"] != "B0":
        certificate = verify_certificate(args.certificate, manifest["dataset"]["revision"], ROOT)
        missing = set(episodes) - set(labels)
        if missing:
            raise RuntimeError(f"G3: {len(missing)} episodes without labels")
    split = episode_split(sorted(episodes), seed=config["training"]["split_seed"])
    train_ids = split["train"][: args.max_episodes or None]
    prop, act = pilot_normalizers(episodes, split["train"], ROOT / "configs")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    backbone = load_local_oft(args.source, args.checkpoint, manifest, config["backbone"]["chunk_size"])
    configure_lora(backbone, config["training"]["lora_rank"])
    policy = JointPolicy(backbone, vcfg["variant"], evidence=vcfg["evidence"], gate=vcfg["gate"],
                         gate_kwargs=vcfg.get("gate_kwargs"), proprio_normalizer=prop, action_normalizer=act).to(args.device)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=config["training"]["learning_rate"])
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    history = []
    for epoch in range(args.epochs):
        order = train_ids[:]
        random.Random(args.seed + epoch).shuffle(order)
        for eid in order:
            h = train_episode(policy, episodes[eid], optimizer, labels.get(eid) if certificate else None,
                              config["training"]["burn_in_calls"], config["training"]["tbptt_calls"], certificate=certificate)
            history.append({"epoch": epoch, "episode": eid, "loss": h[-1]["losses"]})
        save_adapter(policy, dest / f"adapter-epoch{epoch}.pt")
    save_adapter(policy, dest / "adapter.pt")
    prop.save(dest / "proprio-normalizer.json")
    act.save(dest / "action-normalizer.json")
    label_files = [Path(f) / f"{eid.rsplit(':', 1)[1]}.labels.json" for f in args.data for eid in train_ids]
    write(dest / "training.json", {
        "kind": "training_run", "name": vcfg["name"], "variant_config": vcfg, "seed": args.seed, "config": config,
        "config_hash": config_hash(config), "split": split, "trained_episode_ids": train_ids,
        "parameters": parameter_counts(policy), "history": history,
        "provenance": {
            "init_checkpoint": manifest["checkpoint"]["revision"],
            "label_hash": sha256_files([f for f in label_files if f.exists()]) if certificate else "none",
            "train_ids_hash": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
            "chunk_size": config["backbone"]["chunk_size"],
            "tuning_budget": f"{args.epochs} epochs x {len(train_ids)} episodes",
            "code_hash": code_hash(),
            "benchmark_revision": manifest["sources"]["mikasa"]["revision"],
        },
    })


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ["preflight", "synthetic-smoke", "real-data-smoke"]:
        sub.add_parser(cmd).add_argument("--output", default=str(ROOT / f"runs/{cmd}.json"))
    p = sub.add_parser("report")
    p.add_argument("records")
    p.add_argument("--reference", default="KEM")
    p.add_argument("--candidate", default="EEM")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--variant-config", required=True)
    p.add_argument("--data", nargs="+", required=True, help="folders of collected episodes")
    p.add_argument("--source", required=True, help="pinned openvla-oft git checkout")
    p.add_argument("--checkpoint", required=True, help="local HF snapshot named by revision")
    p.add_argument("--certificate")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all train episodes")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cmd == "preflight":
        preflight(args.output)
    elif args.cmd == "synthetic-smoke":
        synthetic_smoke(args.output)
    elif args.cmd == "real-data-smoke":
        real_data_smoke(args.output)
    elif args.cmd == "report":
        records = json.loads(Path(args.records).read_text())
        write(args.output, paired_report(records, args.reference, args.candidate, args.tasks))
    elif args.cmd == "train":
        train(args)


if __name__ == "__main__":
    main()
