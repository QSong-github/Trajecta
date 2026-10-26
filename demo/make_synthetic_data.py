"""
Generate a small synthetic dataset in the exact input format of the Trajecta model.

The values are random and carry no clinical meaning. They only reproduce the
tensor shapes, feature layout and missingness structure expected by the
training and inference scripts.

Outputs (demo/demo_data/):
  sequences.npy          (N, 500, 58) float32   training cohort visit sequences
  masks.npy              (N, 500) bool          True = real visit
  cohort.csv             ID, split (train/val)
  graph_edges.csv        patient_i, patient_j, weight   patient similarity graph
  val_sequences.npy      (M, 500, 58) float32   external validation cohort
  val_masks.npy          (M, 500) bool
  val_cohort.csv         ID
  train_segments.h5ad    obs: ID, seg           trajectory segment of each training patient

Usage:
  python demo/make_synthetic_data.py --n_train 512 --n_val 128
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

T_MAX = 500
N_FEATURES = 58
N_LAB_VITAL = 23          # 19 labs + 4 vitals, each stored as (value, availability)
AVAIL_PROB = np.array([0.60, 0.60, 0.55, 0.15, 0.55, 0.55, 0.50, 0.15, 0.55, 0.50,
                       0.35, 0.50, 0.25, 0.10, 0.35, 0.35, 0.35, 0.35, 0.05,
                       0.70, 0.70, 0.55, 0.30])


def simulate_cohort(n, rng):
    """Return sequences, masks and a per-patient summary used for graph construction."""
    seqs = np.zeros((n, T_MAX, N_FEATURES), dtype=np.float32)
    masks = np.zeros((n, T_MAX), dtype=bool)
    summary = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        T = int(rng.integers(15, 160))
        interval_days = np.concatenate([[0.0], rng.exponential(45.0, T - 1)])
        t_days = np.cumsum(interval_days) + rng.integers(0, 365)
        years = t_days / 365.25

        egfr = np.clip(rng.uniform(20, 100) + rng.normal(-2.0, 3.0) * years
                       + rng.normal(0, 5, T), 5, 130)
        age = rng.uniform(35, 85) + years

        x = np.zeros((T, N_FEATURES), dtype=np.float32)
        # [0:2] time features (robust-scaled): time since CKD diagnosis, interval
        x[:, 0] = (t_days - 1000) / 1000
        x[:, 1] = (interval_days - 30) / 60
        # [2:48] 23 (value, availability) pairs; eGFR is the first pair
        for k in range(N_LAB_VITAL):
            avail = rng.random(T) < AVAIL_PROB[k]
            val = (egfr - 50) / 30 if k == 0 else rng.normal(0, 1, T)
            x[:, 2 + 2 * k] = np.where(avail, val, 0.0)
            x[:, 3 + 2 * k] = avail
        # [48:51] comorbidity flags (diabetes, hypertension, CVD)
        x[:, 48:51] = rng.random(3) < [0.45, 0.80, 0.35]
        # [51] CKD stage 1-5
        x[:, 51] = 5 - np.digitize(egfr, [15, 30, 60, 90])
        # [52] age (scaled), [53] sex, [54:58] race one-hot
        x[:, 52] = (age - 65) / 15
        x[:, 53] = rng.integers(0, 2)
        x[:, 54 + rng.choice(4, p=[0.70, 0.18, 0.04, 0.08])] = 1.0

        seqs[i, :T] = x
        masks[i, :T] = True
        summary[i] = [egfr.mean() / 30, (egfr[-1] - egfr[0]) / max(years[-1], 0.5) / 5, age[0] / 15]
    return seqs, masks, summary


def build_graph(ids, summary, k=15):
    """k-nearest-neighbour patient graph with similarity weights in (0, 1]."""
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(summary)
    dist, idx = nn_.kneighbors(summary)
    seen, rows = set(), []
    for i in range(len(ids)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                rows.append((ids[i], ids[j], float(1.0 / (1.0 + d))))
    return pd.DataFrame(rows, columns=["patient_i", "patient_j", "weight"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=512)
    parser.add_argument("--n_val", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # Training (discovery) cohort
    ids = [f"P{i:05d}" for i in range(args.n_train)]
    seqs, masks, summary = simulate_cohort(args.n_train, rng)
    split = np.where(rng.random(args.n_train) < 0.9, "train", "val")
    np.save(os.path.join(out, "sequences.npy"), seqs)
    np.save(os.path.join(out, "masks.npy"), masks)
    pd.DataFrame({"ID": ids, "split": split}).to_csv(os.path.join(out, "cohort.csv"), index=False)
    edges = build_graph(ids, summary)
    edges.to_csv(os.path.join(out, "graph_edges.csv"), index=False)

    # External validation cohort
    val_ids = [f"V{i:05d}" for i in range(args.n_val)]
    val_seqs, val_masks, _ = simulate_cohort(args.n_val, rng)
    np.save(os.path.join(out, "val_sequences.npy"), val_seqs)
    np.save(os.path.join(out, "val_masks.npy"), val_masks)
    pd.DataFrame({"ID": val_ids}).to_csv(os.path.join(out, "val_cohort.csv"), index=False)

    # Trajectory segment labels of the training patients (6 segments, by mean eGFR)
    import anndata
    seg = 6 - np.digitize(summary[:, 0], np.quantile(summary[:, 0], [1/6, 2/6, 3/6, 4/6, 5/6]))
    obs = pd.DataFrame({"ID": ids, "seg": seg.astype(int)}, index=ids)
    anndata.AnnData(obs=obs).write_h5ad(os.path.join(out, "train_segments.h5ad"))

    print(f"Training cohort:   {args.n_train} patients "
          f"({(split == 'train').sum()} train / {(split == 'val').sum()} val), "
          f"sequences {seqs.shape}, {len(edges)} graph edges")
    print(f"Validation cohort: {args.n_val} patients, sequences {val_seqs.shape}")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
