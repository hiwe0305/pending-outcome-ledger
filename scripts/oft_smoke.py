"""GPU server: real local 7B forward (+ optional backward) for one variant, no training.

Measures peak VRAM and latency for the resource_profile gate. Runs two policy calls
so the intra-chunk evidence path is exercised. Does not change gate statuses.
"""

import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jem.backbone import load_local_oft, configure_lora
from jem.data import Episode, TrainNormalizer
from jem.policy import JointPolicy
from jem.training import parameter_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--variant-config", default=str(ROOT / "configs/variants/EEM.json"))
    p.add_argument("--backward", action="store_true")
    p.add_argument("--output", default=str(ROOT / "runs/oft-smoke.json"))
    args = p.parse_args()
    manifest = json.loads((ROOT / "configs/manifest.json").read_text())
    ep = Episode.load(args.sample)
    if not torch.cuda.is_available():
        raise RuntimeError("Real OFT smoke requires a working CUDA GPU")
    torch.manual_seed(17)
    torch.cuda.reset_peak_memory_stats()
    model = load_local_oft(args.source, args.checkpoint, manifest, for_audit=True).to(
        "cuda"
    )
    if args.backward:
        configure_lora(model)
    prop = TrainNormalizer.load(ROOT / "configs/proprio-normalizer.json")
    act = TrainNormalizer.load(ROOT / "configs/actions-normalizer.json")
    v = json.loads(pathlib.Path(args.variant_config).read_text())
    policy = JointPolicy(model, v["variant"], evidence=v["evidence"], gate=v["gate"], gate_kwargs=v.get("gate_kwargs"),
                         proprio_normalizer=prop, action_normalizer=act).to("cuda")
    policy.eval()
    policy.reset()
    start = time.perf_counter()
    with torch.enable_grad() if args.backward else torch.no_grad():
        from jem.data import calls
        stream = calls(ep, policy.chunk_size)
        for _, call in zip(range(2), stream):
            prediction, output = policy.step(call.observation, call.executed, call.dt, call.frames)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Nonfinite prediction")
        if args.backward:
            prediction.float().square().mean().backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in policy.parameters()
            ):
                raise RuntimeError("Nonfinite gradient")
    torch.cuda.synchronize()
    report = {
        "kind": "real_local_oft_interface_smoke",
        "variant_config": v,
        "backward": args.backward,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "parameters": parameter_counts(policy),
        "action_shape": list(prediction.shape),
        "checkpoint": manifest["checkpoint"],
        "source_revisions": manifest["sources"],
        "episode_id": ep.episode_id,
        "rollout": False,
        "training": False,
        "benchmark_performance": False,
    }
    dest = pathlib.Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
