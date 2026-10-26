"""
Run Trajecta end-to-end on the synthetic demo data:
  1. Stage 1: Mamba self-supervised pre-training      training/01_train_stage1_mamba.py
  2. Stage 2: GCN + gated fusion, embedding export    training/02_train_stage2_gcn_fusion.py
  3. Inference: encode the validation cohort and      inference/assign_validation_segments.py
     assign trajectory segments

Each step calls the unmodified script; only its input/output paths, cohort size
and number of epochs are overridden. Requires a CUDA GPU (mamba-ssm).

Usage:
  python demo/make_synthetic_data.py
  python demo/run_demo.py --epochs 3
"""

import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "demo", "demo_data")
OUT = os.path.join(REPO, "demo", "demo_output")


def load_script(rel_path, name):
    """Import a script as a module without running its main()."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3,
                        help="max epochs per stage (paper: 200 with early stopping)")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(DATA, "sequences.npy")):
        raise SystemExit("Demo data not found. Run: python demo/make_synthetic_data.py")
    n_patients = len(pd.read_csv(os.path.join(DATA, "cohort.csv")))
    stage1_dir, stage2_dir, infer_dir = (os.path.join(OUT, d) for d in ("stage1", "stage2", "inference"))
    timings = {}

    t0 = time.time()
    print(f"\n===== [1/3] Stage 1: Mamba pre-training ({args.epochs} epochs) =====")
    s1 = load_script("training/01_train_stage1_mamba.py", "stage1")
    s1.SEQUENCES_PATH = os.path.join(DATA, "sequences.npy")
    s1.MASKS_PATH = os.path.join(DATA, "masks.npy")
    s1.COHORT_PATH = os.path.join(DATA, "cohort.csv")
    s1.OUTPUT_DIR = stage1_dir
    s1.EPOCHS = args.epochs
    s1.main()
    timings["Stage 1"] = time.time() - t0

    t0 = time.time()
    print(f"\n===== [2/3] Stage 2: GCN + gated fusion ({args.epochs} epochs) =====")
    s2 = load_script("training/02_train_stage2_gcn_fusion.py", "stage2")
    s2.STAGE1_PTH = os.path.join(stage1_dir, "stage1_best.pth")
    s2.SEQUENCES_PATH = os.path.join(DATA, "sequences.npy")
    s2.MASKS_PATH = os.path.join(DATA, "masks.npy")
    s2.COHORT_PATH = os.path.join(DATA, "cohort.csv")
    s2.EDGES_PATH = os.path.join(DATA, "graph_edges.csv")
    s2.OUTPUT_DIR = stage2_dir
    s2.N_PATIENTS = n_patients
    s2.EPOCHS = args.epochs
    s2.main()
    timings["Stage 2"] = time.time() - t0

    t0 = time.time()
    print("\n===== [3/3] Inference: validation cohort segment assignment =====")
    inf = load_script("inference/assign_validation_segments.py", "inference")
    inf.STAGE2_PTH = os.path.join(stage2_dir, "stage2_best.pth")
    inf.TRAIN_SEQ_PATH = os.path.join(DATA, "sequences.npy")
    inf.TRAIN_MASK_PATH = os.path.join(DATA, "masks.npy")
    inf.TRAIN_COHORT_PATH = os.path.join(DATA, "cohort.csv")
    inf.VAL_SEQ_PATH = os.path.join(DATA, "val_sequences.npy")
    inf.VAL_MASK_PATH = os.path.join(DATA, "val_masks.npy")
    inf.VAL_COHORT_PATH = os.path.join(DATA, "val_cohort.csv")
    inf.ADATA_PATH = os.path.join(DATA, "train_segments.h5ad")
    inf.OUTPUT_DIR = infer_dir
    inf.main()
    timings["Inference"] = time.time() - t0

    emb = np.load(os.path.join(stage2_dir, "patient_embedding.npy"))
    print("\n===== Demo finished =====")
    print(f"Patient embeddings: {emb.shape} -> {stage2_dir}")
    print(f"Validation segments: {os.path.join(infer_dir, 'val_seg_assignment.csv')}")
    for k, v in timings.items():
        print(f"  {k:<10s} {v / 60:5.1f} min")


if __name__ == "__main__":
    main()
