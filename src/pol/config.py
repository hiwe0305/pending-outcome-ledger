"""YAML configs with `_base_` inheritance, dotted overrides and run provenance."""

from copy import deepcopy
from pathlib import Path
import hashlib, json

import yaml

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
MIKASA_COMMIT = "3d5a21a99e5df2bcc34c612040b269e394b09438"


def _merge(base, child):
    out = deepcopy(base)
    for k, v in child.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else deepcopy(v)
    return out


def load_config(path, overrides=None):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = _merge(load_config(path.parent / base), cfg)
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return cfg


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def code_hash(code_root):
    h = hashlib.sha256()
    for f in sorted(Path(code_root, "src/pol").rglob("*.py")):
        h.update(f.relative_to(code_root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def provenance(cfg, code_root, extra):
    return {"config_hash": stable_hash(cfg), "code_hash": code_hash(code_root),
            "openpi_commit": OPENPI_COMMIT, "mikasa_commit": MIKASA_COMMIT, **extra}
