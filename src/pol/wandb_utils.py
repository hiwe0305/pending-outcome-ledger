"""Weights & Biases logging shared by the training and evaluation entry points.

Project `pol`; entity from WANDB_ENTITY or the logged-in account. WANDB_MODE=offline or disabled
works unchanged (offline runs are synced later with `wandb sync`). Every run records the resolved
config and its provenance.
"""

import os


class Run:
    def __init__(self, run, log_every):
        self.run, self.log_every = run, log_every

    def log(self, step, metrics, force=False):
        if self.run is not None and (force or step % self.log_every == 0):
            self.run.log({k: float(v) if hasattr(v, "__float__") else v for k, v in metrics.items()}, step=step)

    def summary(self, metrics):
        if self.run is not None:
            for k, v in metrics.items():
                self.run.summary[k] = v

    def finish(self):
        if self.run is not None:
            self.run.finish()


def init_run(stage, variant, cfg, prov, K, seed, log_every=20, project="pol"):
    try:
        import wandb
    except ImportError:
        return Run(None, log_every)
    if os.environ.get("WANDB_MODE") == "disabled":
        return Run(None, log_every)
    kwargs = dict(project=project, entity=os.environ.get("WANDB_ENTITY"), group=f"{stage}/{variant}",
                  name=f"{stage}-{variant}-K{K}-s{seed}", config={**cfg, "provenance": prov}, reinit=True)
    try:
        run = wandb.init(**kwargs)
    except Exception as exc:  # e.g. an invalid API key: never lose a training run to logging
        print(f"[wandb] online init failed ({type(exc).__name__}: {str(exc)[:120]}); logging offline, "
              "run `wandb sync wandb/offline-run-*` after `wandb login`", flush=True)
        run = wandb.init(mode="offline", **kwargs)
    return Run(run, log_every)
