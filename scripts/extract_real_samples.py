"""Read only video byte ranges needed for episode 0, bounded at 24 MB/camera.
Saved NPZ contains RGB (top,wrist), proprio, action and timestamps, no oracle.
"""

import io, json, pathlib, urllib.request
import av, numpy as np, pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]


class RangeFile(io.RawIOBase):
    def __init__(self, url, budget=24_000_000):
        self.url = url
        self.pos = 0
        self.cache = {}
        self.block = 524288
        self.budget = budget
        self.downloaded = 0
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=40
        ) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, n, whence=0):
        self.pos = n if whence == 0 else self.pos + n if whence == 1 else self.size + n
        if self.pos < 0:
            raise ValueError("negative position")
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        out = []
        while n > 0:
            block = self.pos // self.block
            start = block * self.block
            end = min(self.size - 1, start + self.block - 1)
            if block not in self.cache:
                if self.downloaded + end - start + 1 > self.budget:
                    raise RuntimeError("video byte budget exceeded")
                req = urllib.request.Request(
                    self.url + f"?range_block={block}",
                    headers={"Range": f"bytes={start}-{end}"},
                )
                with urllib.request.urlopen(req, timeout=40) as r:
                    if r.status != 206:
                        raise RuntimeError("server ignored Range; refusing full shard")
                    b = r.read(self.block + 1)
                if len(b) != end - start + 1:
                    raise RuntimeError("wrong range size")
                self.cache[block] = b
                self.downloaded += len(b)
            offset = self.pos - start
            take = min(n, len(self.cache[block]) - offset)
            if take <= 0:
                raise RuntimeError("empty video range")
            out.append(self.cache[block][offset : offset + take])
            self.pos += take
            n -= take
        return b"".join(out)


def main():
    revision = json.loads((ROOT / "configs/manifest.json").read_text())["dataset"][
        "revision"
    ]
    report_path = ROOT / "runs/real-sample-extraction.json"
    report_path.parent.mkdir(exist_ok=True)
    reports = json.loads(report_path.read_text()) if report_path.exists() else []
    for folder in sorted((ROOT / "data/audit").glob("batteries*")):
        dest = ROOT / "data/samples" / f"{folder.name}-episode-000000.npz"
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            continue
        meta = (
            pq.read_table(folder / "meta/episodes/chunk-000/file-000.parquet")
            .slice(0, 1)
            .to_pylist()[0]
        )
        rows = pq.read_table(
            folder / "data/chunk-000/file-000.parquet",
            filters=[("episode_index", "=", 0)],
        ).to_pylist()
        cameras = []
        report = {"task_folder": folder.name, "length": len(rows), "video": {}}
        for cam in ("top", "wrist"):
            key = f"observation.images.{cam}"
            ci = meta[f"videos/{key}/chunk_index"]
            fi = meta[f"videos/{key}/file_index"]
            rel = f"{folder.name}/videos/{key}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            url = f"https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot/resolve/{revision}/{rel}"
            f = RangeFile(url)
            frames = []
            pts = []
            with av.open(f, mode="r") as container:
                for frame in container.decode(video=0):
                    t = float(frame.pts * frame.time_base)
                    if t + 1e-4 < meta[f"videos/{key}/from_timestamp"]:
                        continue
                    frames.append(frame.to_ndarray(format="rgb24"))
                    pts.append(t)
                    if len(frames) == len(rows):
                        break
            if len(frames) != len(rows):
                raise RuntimeError("incomplete episode video")
            expected = (
                np.array([r["timestamp"] for r in rows])
                + meta[f"videos/{key}/from_timestamp"]
            )
            err = float(np.max(np.abs(expected - np.array(pts))))
            if err > 0.002:
                raise RuntimeError(f"video sync failed: {err}")
            cameras.append(np.stack(frames))
            report["video"][cam] = {
                "bytes_read": f.downloaded,
                "frames": len(frames),
                "max_pts_error": err,
                "source": rel,
            }
            print(folder.name, cam, report["video"][cam], flush=True)
        np.savez_compressed(
            dest,
            rgb=np.concatenate(cameras, axis=-1),
            proprio=np.array([r["observation.state"] for r in rows], np.float32),
            action=np.array([r["action"] for r in rows], np.float32),
            timestamp=np.array([r["timestamp"] for r in rows], np.float32),
            language=np.array(meta["tasks"][0]),
            episode_id=np.array(f"{folder.name}:0"),
        )
        reports = [r for r in reports if r["task_folder"] != folder.name]
        reports.append(report)
        report_path.write_text(json.dumps(reports, indent=2))
    (ROOT / "runs/real-sample-extraction.json").write_text(
        json.dumps(reports, indent=2)
    )


if __name__ == "__main__":
    main()
