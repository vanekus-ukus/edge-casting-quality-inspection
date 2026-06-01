from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path("experiments/research_20260601/scientific_strengthening/dedup_split_protocol")
DEFAULT_DATA = Path("data/processed")
CLASSES = {"ok": 0, "defective": 1}


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def ahash(path: Path, size: int = 8) -> int:
    image = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | int(p > mean)
    return bits


def hdist(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def collect_images(data_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for cls, label in CLASSES.items():
            for path in sorted((data_dir / split / cls).glob("*")):
                if path.is_file():
                    rows.append(
                        {
                            "path": str(path),
                            "original_split": split,
                            "class": cls,
                            "label": label,
                            "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
                            "ahash": ahash(path),
                        }
                    )
    if not rows:
        raise FileNotFoundError(f"No images found under {data_dir}/{{train,val,test}}/{{ok,defective}}")
    return pd.DataFrame(rows)


def build_groups(df: pd.DataFrame, near_threshold: int) -> pd.DataFrame:
    dsu = DSU(len(df))
    by_sha: dict[str, list[int]] = defaultdict(list)
    by_hash: dict[int, list[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        by_sha[row["sha1"]].append(idx)
        by_hash[int(row["ahash"])].append(idx)
    for indices in by_sha.values():
        for idx in indices[1:]:
            dsu.union(indices[0], idx)
    hashes = list(by_hash)
    for i, ha in enumerate(hashes):
        for hb in hashes[i + 1 :]:
            if hdist(ha, hb) <= near_threshold:
                dsu.union(by_hash[ha][0], by_hash[hb][0])
    df = df.copy()
    df["dedup_group"] = [dsu.find(i) for i in range(len(df))]
    root_to_id = {root: gid for gid, root in enumerate(sorted(df["dedup_group"].unique()))}
    df["dedup_group"] = df["dedup_group"].map(root_to_id)
    return df


def assign_splits(df: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> pd.DataFrame:
    random.seed(seed)
    groups = []
    for gid, g in df.groupby("dedup_group"):
        counts = Counter(g["class"])
        major = counts.most_common(1)[0][0]
        groups.append({"dedup_group": gid, "n": len(g), "major_class": major, "ok": counts["ok"], "defective": counts["defective"]})
    random.shuffle(groups)
    totals = {"train": 0, "val": 0, "test": 0}
    class_totals = {split: Counter() for split in totals}
    target = dict(zip(("train", "val", "test"), ratios))
    assignments = {}
    for group in sorted(groups, key=lambda x: (x["major_class"], -x["n"])):
        best = min(
            totals,
            key=lambda sp: (totals[sp] + group["n"]) / target[sp] + 0.25 * class_totals[sp][group["major_class"]],
        )
        assignments[group["dedup_group"]] = best
        totals[best] += group["n"]
        class_totals[best].update({"ok": group["ok"], "defective": group["defective"]})
    out = df.copy()
    out["clean_split"] = out["dedup_group"].map(assignments)
    return out


def write_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, g in df.groupby("dedup_group"):
        rows.append(
            {
                "dedup_group": gid,
                "n_images": len(g),
                "classes": ",".join(sorted(g["class"].unique())),
                "original_splits": ",".join(sorted(g["original_split"].unique())),
                "clean_split": g["clean_split"].iloc[0],
                "has_cross_split_original": g["original_split"].nunique() > 1,
                "n_exact_sha1": g["sha1"].nunique(),
                "n_unique_ahash": g["ahash"].nunique(),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(ROOT / "dedup_group_stats.csv", index=False)
    df.to_csv(ROOT / "dedup_clean_split_manifest.csv", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "n_images", "value": len(df)},
            {"metric": "n_groups", "value": df["dedup_group"].nunique()},
            {"metric": "cross_split_groups_original", "value": int(stats["has_cross_split_original"].sum())},
            {"metric": "train_images_clean", "value": int((df["clean_split"] == "train").sum())},
            {"metric": "val_images_clean", "value": int((df["clean_split"] == "val").sum())},
            {"metric": "test_images_clean", "value": int((df["clean_split"] == "test").sum())},
        ]
    )
    summary.to_csv(ROOT / "dedup_split_summary.csv", index=False)
    return stats


def materialize(df: pd.DataFrame, target_dir: Path) -> None:
    for _, row in df.iterrows():
        dst = target_dir / row["clean_split"] / row["class"] / Path(row["path"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.74, 0.16, 0.10), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--materialize-dir", type=Path, help="Optional output dir. Never defaults to data/processed.")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    df = collect_images(args.data_dir)
    df = build_groups(df, args.near_threshold)
    df = assign_splits(df, args.seed, tuple(args.ratios))
    stats = write_stats(df)
    if args.materialize_dir:
        if args.materialize_dir.resolve() == args.data_dir.resolve():
            raise ValueError("Refusing to overwrite source data directory.")
        materialize(df, args.materialize_dir)
    print(
        {
            "images": len(df),
            "groups": int(df["dedup_group"].nunique()),
            "cross_split_groups_original": int(stats["has_cross_split_original"].sum()),
            "stats": str(ROOT / "dedup_group_stats.csv"),
            "manifest": str(ROOT / "dedup_clean_split_manifest.csv"),
        }
    )


if __name__ == "__main__":
    main()
