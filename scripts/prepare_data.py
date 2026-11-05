"""Turn SPOCK's dataset zip into the two npz files the benchmarks read.

The download script gives you `data/spock_dataset.zip`. The .bin archives in
there need REBOUND to read; this script extracts them, joins the 1e9 labels,
and writes `data/resonant_real.npz` and `data/random_real.npz`.

Column layout, repeated here because every script depends on it:
    0      star mass
    1..18  3 planets x (m, a, e, pomega, Omega, l)
    19     Stable label from the 1e9 run
    20     instability time

Run it in the kirk environment from the README.
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rebound

ROOT = Path(__file__).resolve().parents[1]


def prepare(split, raw_dir, out_dir, limit=None):
    archive = raw_dir / "dataset" / split / "clean_initial_conditions.bin"
    labels = raw_dir / "dataset" / split / "sim_labels.csv"
    if not archive.exists():
        raise FileNotFoundError(archive)
    if not labels.exists():
        raise FileNotFoundError(labels)

    sa = rebound.Simulationarchive(str(archive))
    lab = pd.read_csv(labels, index_col=0)
    n = len(sa) if limit is None else min(limit, len(sa))
    if len(lab) < n:
        raise ValueError(f"{labels} has {len(lab)} labels but the archive has {n} systems")

    rows = np.zeros((n, 1 + 18 + 2), np.float64)
    for i in range(n):
        sim = sa[i]
        elements = []
        for p in sim.particles[1:]:
            elements += [p.m, p.a, p.e, p.pomega, p.Omega, p.l]
        if len(elements) != 18:
            raise ValueError(f"{split} system {i} has {len(elements) // 6} planets, expected 3")
        rows[i, 0] = sim.particles[0].m
        rows[i, 1:19] = elements
        rows[i, 19] = float(lab["Stable"].iloc[i])
        rows[i, 20] = float(lab["instability_time"].iloc[i])

    out = out_dir / f"{split}_real.npz"
    np.savez(out, X=rows)
    print(f"{split}: {n} systems -> {out} (stable {rows[:, 19].mean():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=ROOT / "data" / "spock_dataset.zip")
    ap.add_argument("--extract-dir", type=Path, default=ROOT / "data" / "spock_raw")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--limit", type=int, help="only read the first N systems per split")
    args = ap.parse_args()

    raw = args.extract_dir
    if not (raw / "dataset").exists():
        if not args.zip.exists():
            raise FileNotFoundError(
                f"{args.zip} not found. Run scripts/download_assets.py first.")
        print(f"extracting {args.zip} -> {raw}")
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("resonant", "random"):
        prepare(split, raw, args.out_dir, args.limit)


if __name__ == "__main__":
    main()
