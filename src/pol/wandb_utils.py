"""Weights & Biases logging shared by the training and evaluation entry points.

Project `pol`; entity from WANDB_ENTITY or the logged-in account. WANDB_MODE=offline or disabled
works unchanged (offline runs are synced later with `wandb sync`). Every run records the resolved
config and its provenance. Each logged step is also printed to stdout and appended to
<out>/train_log.jsonl, so a run can be followed live with `tail -f` without W&B.
"""

import json, os, sys, time


class Run:
    def __init__(self, run, log_every, name="run", jsonl=None):
        self.run, self.log_every, self.name, self.jsonl = run, log_every, name, jsonl
        self.t0 = time.time()

    def log(self, step, metrics, force=False):
        if not (force or step % self.log_every == 0):
            return
        clean = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items() if v is not None}
        if self.run is not None:
            self.run.log(clean, step=step)
        shown = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in clean.items())
        print(f"[{self.name}] step {step} t={time.time() - self.t0:.0f}s {shown}", file=sys.stdout, flush=True)
        if self.jsonl is not None:
            with open(self.jsonl, "a") as f:
                f.write(json.dumps({"step": step, "time": round(time.time() - self.t0, 2), **clean}) + "\n")

    def summary(self, metrics):
        if self.run is not None:
            for k, v in metrics.items():
                self.run.summary[k] = v

    def finish(self):
        if self.run is not None:
            self.run.finish()


def init_run(stage, variant, cfg, prov, K, seed, log_every=20, project="pol", out_dir=None):
    name = f"{stage}-{variant}-K{K}-s{seed}"
    jsonl = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        jsonl = os.path.join(out_dir, "train_log.jsonl")
        open(jsonl, "w").close()
    try:
        import wandb
    except ImportError:
        return Run(None, log_every, name, jsonl)
    if os.environ.get("WANDB_MODE") == "disabled":
        return Run(None, log_every, name, jsonl)
    kwargs = dict(project=project, entity=os.environ.get("WANDB_ENTITY"), group=f"{stage}/{variant}",
                  name=name, config={**cfg, "provenance": prov}, reinit=True)
    try:
        run = wandb.init(**kwargs)
    except Exception as exc:  # e.g. an invalid API key: never lose a training run to logging
        print(f"[wandb] online init failed ({type(exc).__name__}: {str(exc)[:120]}); logging offline, "
              "run `wandb sync wandb/offline-run-*` after `wandb login`", flush=True)
        run = wandb.init(mode="offline", **kwargs)
    return Run(run, log_every, name, jsonl)
