"""
Validation Inference (Mamba-only version)

No GCN and no graph: Mamba encodes each patient independently.
Mamba embeddings are also re-extracted for the training cohort (as the KNN reference).

Pipeline:
  Step 1: Load the frozen Stage2 model (Mamba part only)
  Step 2: Mamba-encode 8,192 training patients → 128D (KNN reference)
  Step 3: Mamba-encode 2,240 validation patients → 128D
  Step 4: KNN segment assignment (val → train Mamba space)
  Step 5: Save

Required files:
  stage2_best.pth
  sequences.npy + masks.npy (training)
  val_sequences.npy + val_masks.npy (validation)
  cohort_8192_split.csv
  val_cohort.csv
  adata_patient_tree_2d_noslope.h5ad (segment labels)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.neighbors import NearestNeighbors

# ============================================================
# CONFIG
# ============================================================
STAGE2_PTH = "./8192_race_fixed/finetune_results/3.0noslope/stage2_best.pth"

TRAIN_SEQ_PATH    = "./8192_race_fixed/sequences.npy"
TRAIN_MASK_PATH   = "./8192_race_fixed/masks.npy"
TRAIN_COHORT_PATH = "./8192_race_fixed/cohort_8192_split.csv"

VAL_SEQ_PATH    = "./validation/val_sequences_fixed.npy"
VAL_MASK_PATH   = "./validation/val_masks.npy"
VAL_COHORT_PATH = "./validation/cohort_validation.csv"

ADATA_PATH = "./figures/patient_level/adata_patient_tree_2d_noslope.h5ad"

OUTPUT_DIR = "./validation/results_mamba_only/"

SEG_KNN_K  = 15
BATCH_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Model definition (only the Mamba part is needed, but the full state_dict is loaded)
# ============================================================

D_INPUT = 51
D_MODEL = 128
D_PROJ  = 64
D_MAE   = 23

from mamba_ssm import Mamba
from torch_geometric.nn import GCNConv


class SinusoidalTimeEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear  = nn.Linear(d_model * 2, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def _encode(self, t):
        B, T, _ = t.shape
        d   = self.d_model
        div = torch.exp(
            torch.arange(0, d, 2, device=t.device).float()
            * (-np.log(10000.0) / d)
        )
        pe = torch.zeros(B, T, d, device=t.device)
        pe[:, :, 0::2] = torch.sin(t * div)
        pe[:, :, 1::2] = torch.cos(t * div)
        return pe

    def forward(self, x_time):
        t_abs    = x_time[:, :, 0:1] / 365.25
        t_rel    = x_time[:, :, 1:2] / 365.25
        emb_abs  = self._encode(t_abs)
        emb_rel  = self._encode(t_rel)
        combined = torch.cat([emb_abs, emb_rel], dim=-1)
        return self.norm(self.linear(combined))


class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class Stage2Model(nn.Module):
    def __init__(self, d_input=D_INPUT, d_model=D_MODEL,
                 d_proj=D_PROJ, d_mae=D_MAE, n_mamba_layers=4):
        super().__init__()
        self.input_proj   = nn.Linear(d_input, d_model)
        self.input_norm   = nn.LayerNorm(d_model)
        self.time_encoder = SinusoidalTimeEncoder(d_model)
        self.mamba_blocks = nn.ModuleList(
            [MambaBlock(d_model) for _ in range(n_mamba_layers)]
        )
        self.mae_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_mae),
        )
        self.egfr_head    = nn.Linear(d_model, 1)
        self.gcn1         = GCNConv(d_model, d_model)
        self.gcn1_norm    = nn.LayerNorm(d_model)
        self.gcn2         = GCNConv(d_model, d_model)
        self.gcn2_norm    = nn.LayerNorm(d_model)
        self.gate_linear  = nn.Linear(d_model * 2, d_model)
        self.proj_head    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

    def encode(self, x, mask):
        """Mamba encode only — does not touch the GCN"""
        x_feat = x[:, :, 2:53]
        x_time = x[:, :, 0:2]
        h = self.input_norm(
            self.input_proj(x_feat) + self.time_encoder(x_time)
        )
        for block in self.mamba_blocks:
            h = block(h)
        mask_f  = mask.unsqueeze(-1).float()
        pat_emb = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
        return pat_emb


# ============================================================
# Mamba encode (batched inference)
# ============================================================

@torch.no_grad()
def mamba_encode(model, sequences, masks, label=""):
    N = sequences.shape[0]
    pat_list = []
    for i in range(0, N, BATCH_SIZE):
        end    = min(i + BATCH_SIZE, N)
        x_b    = sequences[i:end].to(DEVICE)
        mask_b = masks[i:end].to(DEVICE)
        with autocast():
            pat_emb = model.encode(x_b, mask_b)
        pat_list.append(pat_emb.cpu().float())
    result = torch.cat(pat_list, dim=0)
    print(f"  {label}: {result.shape}")
    return result


# ============================================================
# KNN segment assignment
# ============================================================

def assign_segments(val_emb, train_emb, train_segments, k=SEG_KNN_K):
    print(f"\n[Step 4] KNN segment assignment (K={k}) in Mamba space...")

    knn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(train_emb)
    distances, indices = knn.kneighbors(val_emb)

    segments    = []
    confidences = []
    unique_segs = sorted(set(train_segments))

    for i in range(val_emb.shape[0]):
        neighbor_segs = train_segments[indices[i]]
        weights = 1.0 / (distances[i] + 1e-8)

        vote_scores = {}
        for seg in unique_segs:
            mask = (neighbor_segs == seg)
            vote_scores[seg] = weights[mask].sum()

        total = sum(vote_scores.values())
        best  = max(vote_scores, key=vote_scores.get)

        segments.append(best)
        confidences.append(vote_scores[best] / total)

    return np.array(segments), np.array(confidences)


# ============================================================
# Sanity check: LOO-KNN on training
# ============================================================

def sanity_check(train_emb, train_segments, k=SEG_KNN_K):
    print(f"\n[Sanity] LOO-KNN on training Mamba embeddings (K={k})...")
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    knn.fit(train_emb)
    _, indices = knn.kneighbors(train_emb)

    correct = 0
    for i in range(len(train_emb)):
        nbr_segs = train_segments[indices[i, 1:k+1]]
        vals, cnts = np.unique(nbr_segs, return_counts=True)
        if vals[cnts.argmax()] == train_segments[i]:
            correct += 1

    acc = correct / len(train_emb) * 100
    print(f"  LOO accuracy: {correct}/{len(train_emb)} = {acc:.1f}%")
    return acc


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Model ──────────────────────────────────────────────
    print("[Step 1] Loading model...")
    model = Stage2Model().to(DEVICE)
    model.load_state_dict(torch.load(STAGE2_PTH, map_location=DEVICE), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  ✅ Model loaded")

    # ── 2. Training Mamba encode ─────────────────────────────
    print("\n[Step 2] Training Mamba encode...")
    train_seq  = torch.from_numpy(np.load(TRAIN_SEQ_PATH).astype(np.float32))
    train_mask = torch.from_numpy(np.load(TRAIN_MASK_PATH))
    train_emb  = mamba_encode(model, train_seq, train_mask, label="Training")

    # Free training sequences to save memory
    del train_seq, train_mask

    # Load segment labels
    import scanpy as sc
    adata = sc.read_h5ad(ADATA_PATH)
    seg_col = None
    for col in ['segment', 'seg', 'Segment', 'tree_segment']:
        if col in adata.obs.columns:
            seg_col = col
            break
    if seg_col is None:
        print(f"  ⚠️  adata.obs columns: {list(adata.obs.columns)}")
        raise ValueError("Segment column not found")
    train_segments = adata.obs[seg_col].values.astype(int)
    print(f"  Segments: {np.unique(train_segments)}")

    # Sanity check
    sanity_check(train_emb.numpy(), train_segments)

    # Save training Mamba embeddings (avoids recomputation later)
    np.save(os.path.join(OUTPUT_DIR, "train_mamba_embedding.npy"), train_emb.numpy())

    # ── 3. Validation Mamba encode ───────────────────────────
    print("\n[Step 3] Validation Mamba encode...")
    val_seq    = torch.from_numpy(np.load(VAL_SEQ_PATH).astype(np.float32))
    val_mask   = torch.from_numpy(np.load(VAL_MASK_PATH))
    val_cohort = pd.read_csv(VAL_COHORT_PATH)
    val_emb    = mamba_encode(model, val_seq, val_mask, label="Validation")
    del val_seq, val_mask

    # ── 4. Segment assignment ────────────────────────────────
    segments, confidences = assign_segments(
        val_emb.numpy(), train_emb.numpy(), train_segments)

    # ── 5. Save ──────────────────────────────────────────────
    print("\n[Step 5] Saving...")
    result_df = pd.DataFrame({
        'patient_id':     val_cohort['ID'].values,
        'segment':        segments,
        'knn_confidence': np.round(confidences, 4),
    })
    result_df.to_csv(os.path.join(OUTPUT_DIR, "val_seg_assignment.csv"), index=False)
    np.save(os.path.join(OUTPUT_DIR, "val_mamba_embedding.npy"), val_emb.numpy())

    # ── Summary ──────────────────────────────────────────────────
    N_train = len(train_segments)
    N_val   = len(segments)

    print("\n" + "=" * 60)
    print("✅ Done (Mamba-only)")
    print("=" * 60)

    print(f"\n{'Seg':<6} {'Training':>14} {'Validation':>14}")
    print("-" * 36)
    for seg in sorted(np.unique(train_segments)):
        nt = (train_segments == seg).sum()
        nv = (segments == seg).sum()
        print(f"  {seg:<4} {nt:>5} ({nt/N_train*100:5.1f}%)  "
              f"{nv:>5} ({nv/N_val*100:5.1f}%)")

    print(f"\nMean KNN confidence: {confidences.mean():.3f}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()