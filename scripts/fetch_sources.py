"""Fetch selected exact-revision audit source; never execute downloaded code."""

import concurrent.futures, json, pathlib, urllib.request, hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOS = {
    "mikasa": "CognitiveAISystems/MIKASA-Robo",
    "oft": "moojink/openvla-oft",
    "memoryvla": "shihao1895/MemoryVLA",
}


def main():
    (ROOT / "runs").mkdir(exist_ok=True)
    jobs = []
    manifest = {}
    for name, repo in REPOS.items():
        sha = json.loads((ROOT / "configs/manifest.json").read_text())["sources"][name][
            "revision"
        ]
        manifest[name] = {
            "repository": repo,
            "revision": sha,
            "scope": "selected audit sources, not complete install",
        }
        if name == "memoryvla":
            paths = ["README.md"]
        else:
            tree_path = ROOT / f"runs/{name}-tree.json"
            if not tree_path.exists():
                with urllib.request.urlopen(
                    f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1",
                    timeout=30,
                ) as r:
                    tree_path.write_bytes(r.read(5_000_000))
            tree = json.loads(tree_path.read_text())["tree"]
            paths = [
                x["path"]
                for x in tree
                if x["type"] == "blob"
                and (
                    x["path"]
                    in [
                        "README.md",
                        "LICENSE",
                        "pyproject.toml",
                        "mikasa_robo_vla_envs.csv",
                        "examples/eval_demo.py",
                        "SETUP.md",
                    ]
                    or (
                        name == "mikasa"
                        and (
                            x["path"].startswith("mikasa_robo_suite/vla/")
                            and x["path"].endswith(".py")
                            or x["path"].startswith("docs/source/")
                            and x["path"].endswith(".rst")
                            and any(
                                s in x["path"]
                                for s in [
                                    "observation_space",
                                    "evaluation_protocol",
                                    "batteries_checker",
                                ]
                            )
                        )
                    )
                    or (
                        name == "oft"
                        and x["path"]
                        in [
                            "prismatic/extern/hf/modeling_prismatic.py",
                            "prismatic/models/action_heads.py",
                            "prismatic/models/projectors.py",
                            "prismatic/vla/constants.py",
                            "prismatic/training/train_utils.py",
                            "experiments/robot/openvla_utils.py",
                            "vla-scripts/finetune.py",
                        ]
                    )
                )
            ]
        for p in paths:
            jobs.append((name, repo, sha, p))

    def get(job):
        name, repo, sha, p = job
        dest = ROOT / "third_party" / name / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with urllib.request.urlopen(
                f"https://raw.githubusercontent.com/{repo}/{sha}/{p}", timeout=30
            ) as r:
                b = r.read(2_000_001)
            if len(b) > 2_000_000:
                raise RuntimeError("source file budget exceeded")
            dest.write_bytes(b)
        return (
            str(dest.relative_to(ROOT)),
            hashlib.sha256(dest.read_bytes()).hexdigest(),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        manifest["files"] = dict(pool.map(get, jobs))
    (ROOT / "third_party/revisions.json").write_text(json.dumps(manifest, indent=2))
    print("Fetched", len(jobs), "source files")


if __name__ == "__main__":
    main()
