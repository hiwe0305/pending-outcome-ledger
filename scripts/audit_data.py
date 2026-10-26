"""Verify pinned parquet/metadata and extracted RGB samples, save split IDs."""

import hashlib, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from jem.data import Episode, TASKS, episode_split, TrainNormalizer
import numpy as np, pyarrow.parquet as pq
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    reports = []
    all_ids = []
    norm_episodes = {}
    for task, folder in TASKS.items():
        base = ROOT / "data/audit" / folder
        info = json.loads((base / "meta/info.json").read_text())
        source = json.loads((base / "source_rlds_metadata.json").read_text())
        episodes = pq.read_table(
            base / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pylist()
        rows = pq.read_table(base / "data/chunk-000/file-000.parquet").to_pylist()
        table = pq.read_table(base / "meta/tasks.parquet").to_pylist()
        sample = Episode.load(ROOT / f"data/samples/{folder}-episode-000000.npz")
        assert (
            source["env_id"] == task and info["total_episodes"] == len(episodes) == 250
        )
        assert (
            table[0]["task_index"] == 0
            and table[0]["__index_level_0__"] == sample.language
        )
        first = [r for r in rows if r["episode_index"] == 0]
        assert len(first) == episodes[0]["length"] == len(sample.actions)
        assert np.array_equal(
            np.array([r["frame_index"] for r in first]), np.arange(len(first))
        )
        assert np.allclose(sample.actions, np.array([r["action"] for r in first]))
        assert np.allclose(
            sample.proprio, np.array([r["observation.state"] for r in first])
        )
        assert np.allclose(np.diff(sample.timestamps), 0.1, atol=2e-5)
        assert (
            sample.rgb.shape[-1] == 6
            and np.std(sample.rgb[..., :3]) > 0
            and np.std(sample.rgb[..., 3:]) > 0
        )
        by_episode = {}
        for row in rows:
            by_episode.setdefault(row["episode_index"], []).append(row)
        for ep in episodes:
            eid = f"{folder}:{ep['episode_index']}"
            all_ids.append(eid)
            er = by_episode.get(ep["episode_index"])
            if er is None:
                continue
            assert len(er) == ep["length"]
            norm_episodes[eid] = SimpleNamespace(
                proprio=np.array([r["observation.state"] for r in er], np.float32),
                actions=np.array([r["action"] for r in er], np.float32),
            )
        reports.append(
            {
                "task": task,
                "folder": folder,
                "sample_episode_id": sample.episode_id,
                "sample_frames": len(first),
                "dataset_episodes": len(episodes),
                "instruction": sample.language,
                "first_source_seed": source["episode_seeds"][0],
                "seed_replay_verified": False,
                "parquet_columns": list(rows[0]),
                "camera_order": ["top", "wrist"],
                "fps": info["fps"],
                "original_labels": ["action"],
                "LQE_labels": False,
                "sample_sha256": hashlib.sha256(
                    (ROOT / f"data/samples/{folder}-episode-000000.npz").read_bytes()
                ).hexdigest(),
            }
        )
    split = episode_split(all_ids, 17)
    (ROOT / "configs/episode-splits.json").write_text(
        json.dumps({"seed": 17, **split}, indent=2)
    )
    # Bounded shards happen to include all 250 numeric trajectories/task. No video download needed to fit numeric statistics.
    if set(split["train"]) <= set(norm_episodes):
        for field in ["actions", "proprio"]:
            TrainNormalizer.fit(norm_episodes, split["train"], field).save(
                ROOT / f"configs/{field}-normalizer.json"
            )
    else:
        raise RuntimeError(
            "Train split missing numeric episodes; cannot fit normalizer"
        )
    (ROOT / "runs/data-audit.json").write_text(
        json.dumps(
            {
                "kind": "real_data_audit",
                "tasks": reports,
                "split_counts": {k: len(v) for k, v in split.items()},
                "labels_gate": "G3 pending",
                "numeric_normalization": "fit only saved train IDs; no dataset-wide stats reused",
            },
            indent=2,
        )
    )
    print(
        "Verified four real episodes; 1000 episode IDs split into",
        {k: len(v) for k, v in split.items()},
    )


if __name__ == "__main__":
    main()
