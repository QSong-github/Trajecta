"""
Step1: Stage 1 pretraining -- Mamba temporal pretraining (no GCN, no gate fusion)

Goal:
  Let Mamba focus solely on learning longitudinal EHR sequences and output stable, high-quality visit/patient embeddings,
  providing good initial features for GCN fine-tuning in Step2.

Model architecture:
  Input Encoding (feature projection + sinusoidal time encoding)
  → Mamba SSM × 4
  → Masked Mean Pool
  → [main tasks] MAE reconstruction head / eGFR prediction head
  → [contrastive task] projection head (proj_head, two independent forward passes)

Training strategy:
  - 3 forward passes: Pass1 (clean input, main tasks), Pass2+3 (each independently augmented, contrastive task)
  - Early Stopping: monitors val_egfr_loss only, not val_mae_loss
  - 9:1 Train/Val split (transductive; the split only affects which samples contribute to the loss)

Input files:
  - sequences.npy               (8192, 500, 58)
  - masks.npy                   (8192, 500)
  - cohort_8192_split.csv       contains split column (train / val)

Output files:
  - stage1_best.pth             model weights with the best validation eGFR loss
  - stage1_train_log.csv        per-epoch train_loss / val_egfr / val_mae
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from mamba_ssm import Mamba

# ============================================================
# Path configuration
# ============================================================
DATA_DIR    = "./8192_race_fixed/"
OUTPUT_DIR  = "./8192_race_fixed/mamba_pretrain_results/3.0noslope/"
COHORT_PATH = "./8192_race_fixed/cohort_8192_split.csv"

SEQUENCES_PATH = os.path.join(DATA_DIR, "sequences.npy")
MASKS_PATH     = os.path.join(DATA_DIR, "masks.npy")

# ============================================================
# Hyperparameters
# ============================================================
# Drop sex(idx53) + race_01~04(idx54~57), keep age(idx52)
# Before: x[:,:,2:]    → 56 columns (all features)
# Now:    x[:,:,2:53]  → 51 columns (sex and race removed)
D_INPUT      = 51       # 58 - 2 time - 5 demographic (sex+race×4) = 51
D_MODEL      = 128
D_PROJ       = 64
D_MAE_OUTPUT = 23
T_MAX        = 500

EGFR_VAL_IDX = 0        # position of EGFR_val in x_feat
                        # x_feat = x[:,:,2:53], so EGFR_val (original col2) is now idx=0

BATCH_SIZE   = 128
LR           = 1e-3
EPOCHS       = 200
TEMPERATURE  = 0.07

DROP_VISIT_RATIO   = 0.15
DROP_FEATURE_RATIO = 0.15

# Loss weights
W_MAE      = 0.6
W_EGFR     = 0.3
W_CONTRAST = 0.1
W_SMOOTH   = 0.5

# Early Stopping
PATIENCE   = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# Model components
# ============================================================

class SinusoidalTimeEncoder(nn.Module):
    """
    Encode time scalars into D_MODEL-dimensional vectors.
    Absolute time (years since CKD diagnosis) and relative time (years since previous visit) are encoded separately,
    concatenated, and projected back to D_MODEL dimensions.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear  = nn.Linear(d_model * 2, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def _encode(self, t):
        """t: (B, T, 1) → (B, T, D_MODEL)"""
        B, T, _ = t.shape
        d = self.d_model
        div = torch.exp(
            torch.arange(0, d, 2, device=t.device).float()
            * (-np.log(10000.0) / d)
        )                               # (d//2,)
        pe = torch.zeros(B, T, d, device=t.device)
        pe[:, :, 0::2] = torch.sin(t * div)
        pe[:, :, 1::2] = torch.cos(t * div)
        return pe                       # (B, T, D_MODEL)

    def forward(self, x_time):
        """
        x_time: (B, T, 2)  col0=time_since_CKD_dx, col1=interval (both /365.25)
        returns: (B, T, D_MODEL)
        """
        t_abs = x_time[:, :, 0:1] / 365.25   # (B, T, 1) → years
        t_rel = x_time[:, :, 1:2] / 365.25   # (B, T, 1) → years

        emb_abs = self._encode(t_abs)          # (B, T, D_MODEL)
        emb_rel = self._encode(t_rel)          # (B, T, D_MODEL)

        combined = torch.cat([emb_abs, emb_rel], dim=-1)  # (B, T, 2*D_MODEL)
        return self.norm(self.linear(combined))            # (B, T, D_MODEL)


class MambaBlock(nn.Module):
    """
    Single Mamba block: Pre-Norm + Mamba SSM + residual connection
    h_out = h + Mamba(LayerNorm(h))
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class Stage1Model(nn.Module):
    """
    Full Stage 1 model (Mamba only, no GCN, no gate fusion)

    Forward flow:
      x (B, T, 58)
      → split: x_feat (B,T,56) + x_time (B,T,2)
      → input_proj(x_feat) + time_encoder(x_time) → (B,T,128)
      → MambaBlock × 4 → visit_emb (B,T,128)
      → masked_mean_pool → pat_emb (B,128)
    """
    def __init__(self, d_input=D_INPUT, d_model=D_MODEL, d_proj=D_PROJ,
                 d_mae=D_MAE_OUTPUT, n_mamba_layers=4):
        super().__init__()

        # Input projection
        self.input_proj   = nn.Linear(d_input, d_model)
        self.input_norm   = nn.LayerNorm(d_model)
        self.time_encoder = SinusoidalTimeEncoder(d_model)

        # Mamba backbone
        self.mamba_blocks = nn.ModuleList(
            [MambaBlock(d_model) for _ in range(n_mamba_layers)]
        )

        # Main-task decoder heads
        self.mae_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_mae),
        )
        self.egfr_head = nn.Linear(d_model, 1)

        # Contrastive projection head (Stage1 only; retrained in Step2)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

    def encode(self, x, mask):
        """
        x:    (B, T, 58)
        mask: (B, T)  True=real visit

        Returns:
          visit_emb: (B, T, D_MODEL)
          pat_emb:   (B, D_MODEL)  masked mean pool
        """
        # Drop sex(col53) + race_01~04(col54~57), keep age(col52)
        x_feat = x[:, :, 2:53]        # (B, T, 51)  ← previously x[:,:,2:]
        x_time = x[:, :, 0:2]         # (B, T, 2)

        # Feature projection + time encoding → summed
        h = self.input_norm(
            self.input_proj(x_feat) + self.time_encoder(x_time)
        )                               # (B, T, D_MODEL)

        # Mamba × 4
        for block in self.mamba_blocks:
            h = block(h)                # (B, T, D_MODEL)

        visit_emb = h                   # (B, T, D_MODEL)

        # Masked mean pool (average over real visits only, ignoring padding)
        mask_f = mask.unsqueeze(-1).float()           # (B, T, 1)
        pat_emb = (visit_emb * mask_f).sum(dim=1) \
                  / mask_f.sum(dim=1).clamp(min=1e-8) # (B, D_MODEL)

        return visit_emb, pat_emb

    def forward(self, x, mask):
        return self.encode(x, mask)


# ============================================================
# Data augmentation (for contrastive learning)
# ============================================================

def augment_drop_visits(x, mask, ratio=DROP_VISIT_RATIO):
    """
    View1: randomly drop a `ratio` fraction of visit rows
    Drop = zero the whole row + set mask to False
    Only real visit rows (mask=True) are dropped
    """
    x_aug    = x.clone()
    mask_aug = mask.clone()

    # Sample independently for each sample
    drop_prob = torch.rand_like(mask.float())           # (B, T)
    drop_mask = (drop_prob < ratio) & mask              # drop real visits only

    x_aug[drop_mask]    = 0.0
    mask_aug[drop_mask] = False

    return x_aug, mask_aug


def augment_drop_features(x, mask, ratio=DROP_FEATURE_RATIO):
    """
    View2: visit-level feature dropout
    For each visit of each patient, independently decide whether to drop each measurement
    Drop = zero that visit's _val + the corresponding _avail

    Granularity: (B, T, 19)
    Meaning: "patient A had no creatinine drawn at visit 23", rather than "all creatinine values of the patient are missing"
    Only real visit rows (mask=True) are modified; padding rows are already 0
    """
    x_aug = x.clone()
    B, T, F = x.shape
    n_val_features = 23   # labs (19) + vitals (4) = 23 val/avail pairs
                          # indices span [02]EGFR to [47]HT; BP/WT/HT must not be omitted

    # Visit-level mask: sampled independently per patient, per visit, per feature
    drop = torch.rand(B, T, n_val_features, device=x.device) < ratio  # (B, T, 19)

    # Only operate on real visit rows (padding rows untouched to avoid adding noise)
    drop = drop & mask.unsqueeze(-1)   # (B, T, 19), padding positions forced to False

    for feat_i in range(n_val_features):
        val_col   = 2 + feat_i * 2    # column index in the 58-dim input (skipping the first 2 time columns)
        avail_col = val_col + 1
        if val_col >= F or avail_col >= F:
            break
        visit_mask = drop[:, :, feat_i]          # (B, T)
        x_aug[:, :, val_col]  [visit_mask] = 0.0
        x_aug[:, :, avail_col][visit_mask] = 0.0

    return x_aug, mask


# ============================================================
# Loss functions
# ============================================================

def mae_loss(model, visit_emb, x_feat_raw):
    """
    MAE reconstruction loss: predict the 23 _val columns, computed only where avail=1

    visit_emb:   (B, T, D_MODEL)
    x_feat_raw:  (B, T, 56)  raw features (time columns not removed, but here already x[:,2:])

    Indices of _val columns in x_feat_raw: 0,2,4,...,44 (23 total)
    Indices of _avail columns:             1,3,5,...,45
    """
    pred = model.mae_decoder(visit_emb)   # (B, T, 23)

    val_indices   = list(range(0, 46, 2))[:23]   # [0,2,4,...,44]
    avail_indices = list(range(1, 47, 2))[:23]   # [1,3,5,...,45]

    target = x_feat_raw[:, :, val_indices]        # (B, T, 23)
    avail  = x_feat_raw[:, :, avail_indices]      # (B, T, 23)

    # Compute loss only where avail=1
    valid_mask = avail > 0.5
    if valid_mask.sum() == 0:
        return pred.sum() * 0.0   # keep the graph connected, zero gradient

    loss = F.mse_loss(pred[valid_mask], target[valid_mask])
    return loss


def egfr_loss(model, visit_emb, x_feat_raw, mask):
    """
    eGFR prediction loss: predict eGFR at t+1 from time t, computed only where t+1 has avail=1

    Index of EGFR_val in x_feat_raw: EGFR_VAL_IDX = 2
    Index of EGFR_avail:             3
    """
    # Use embeddings of the first T-1 steps to predict eGFR of the last T-1 steps
    pred   = model.egfr_head(visit_emb[:, :-1, :]).squeeze(-1)  # (B, T-1)
    target = x_feat_raw[:, 1:, EGFR_VAL_IDX]                    # (B, T-1)
    avail  = x_feat_raw[:, 1:, EGFR_VAL_IDX + 1]                # (B, T-1)

    # mask: next step must be a real visit (mask=True) with an eGFR record (avail=1)
    next_mask = mask[:, 1:] & (avail > 0.5)

    if next_mask.sum() == 0:
        return pred.sum() * 0.0   # keep the graph connected, zero gradient

    loss = F.mse_loss(pred[next_mask], target[next_mask])
    return loss


def smooth_loss(visit_emb, mask, x_time_raw):
    """
    Time-aware smoothness loss:
    embeddings of adjacent visits should point in similar directions; shorter intervals are penalized more

    Formula: L = Σ (1 - cosine_sim(emb[t], emb[t-1])) × exp(-interval/90)

    x_time_raw: (B, T, 2), col1=interval (raw value, in days)
    """
    # Cosine similarity between adjacent steps
    e1 = visit_emb[:, :-1, :]   # (B, T-1, D_MODEL)
    e2 = visit_emb[:, 1:,  :]   # (B, T-1, D_MODEL)
    cos_sim = F.cosine_similarity(e1, e2, dim=-1)      # (B, T-1)

    # interval is in raw days; decay weight with a 90-day scale
    interval = x_time_raw[:, 1:, 1]                    # (B, T-1)
    weight   = torch.exp(-interval.abs() / 90.0)       # (B, T-1)

    # Compute only where both steps are real visits
    valid = mask[:, :-1] & mask[:, 1:]                 # (B, T-1)
    if valid.sum() == 0:
        return cos_sim.sum() * 0.0   # keep the graph connected, zero gradient

    loss = ((1.0 - cos_sim) * weight)[valid].mean()
    return loss


def contrastive_loss_stage1(model, x, mask):
    """
    NT-Xent contrastive loss (Stage 1, Mamba only, no GCN)

    Fixes:
      - Fix1: labels have length 2B (first B match last B, last B match first B)
      - Mask the diagonal (a sample is not its own positive)
    """
    B = x.shape[0]

    # Two independently perturbed views
    view1, mask1 = augment_drop_visits(x, mask)
    view2, mask2 = augment_drop_features(x, mask)

    # Each through Mamba → pool → projection
    _, pat_v1 = model.encode(view1, mask1)             # (B, D_MODEL)
    _, pat_v2 = model.encode(view2, mask2)             # (B, D_MODEL)

    z1 = F.normalize(model.proj_head(pat_v1), dim=-1)  # (B, D_PROJ)
    z2 = F.normalize(model.proj_head(pat_v2), dim=-1)  # (B, D_PROJ)

    # Concatenate into (2B, D_PROJ) and compute the similarity matrix
    z   = torch.cat([z1, z2], dim=0)                   # (2B, D_PROJ)
    sim = z @ z.T / TEMPERATURE                        # (2B, 2B)

    # Mask the diagonal (self is not a positive)
    sim.fill_diagonal_(float('-inf'))

    # Fix2: full 2B labels
    # positive of z1[i] is z2[i], at position i+B
    # positive of z2[i] is z1[i], at position i
    labels = torch.cat([
        torch.arange(B, 2 * B, device=x.device),
        torch.arange(0, B,     device=x.device),
    ])                                                  # (2B,)

    loss = F.cross_entropy(sim, labels)
    return loss


# ============================================================
# Training and validation
# ============================================================

def train_one_epoch(model, loader, sequences, masks, optimizer, scaler):
    model.train()
    total_loss = total_mae = total_egfr = 0.0
    total_cont = total_smooth = 0.0
    n_batches = 0

    for (batch_idx,) in loader:
        x    = sequences[batch_idx].to(DEVICE)
        mask = masks[batch_idx].to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            # ── Pass 1: clean input → main tasks ──────────────────────────
            visit_emb_clean, _ = model.encode(x, mask)

            x_feat_raw = x[:, :, 2:53]   # (B, T, 51) sex/race removed
            x_time_raw = x[:, :, 0:2]

            L_mae    = mae_loss(model, visit_emb_clean, x_feat_raw)
            L_egfr   = egfr_loss(model, visit_emb_clean, x_feat_raw, mask)
            L_smooth = smooth_loss(visit_emb_clean, mask, x_time_raw)

            # ── Pass 2 & 3: contrastive task ──────────────────────────────
            L_contrast = contrastive_loss_stage1(model, x, mask)

            loss = (W_MAE      * L_mae     +
                    W_EGFR     * L_egfr    +
                    W_CONTRAST * L_contrast +
                    W_SMOOTH   * L_smooth)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss   += loss.item()
        total_mae    += L_mae.item()
        total_egfr   += L_egfr.item()
        total_cont   += L_contrast.item()
        total_smooth += L_smooth.item()
        n_batches    += 1

    return {
        'train_loss':     total_loss   / n_batches,
        'train_mae':      total_mae    / n_batches,
        'train_egfr':     total_egfr   / n_batches,
        'train_contrast': total_cont   / n_batches,
        'train_smooth':   total_smooth / n_batches,
    }


@torch.no_grad()
def validate(model, loader, sequences, masks):
    model.eval()
    val_egfr_list = []
    val_mae_list  = []

    for (batch_idx,) in loader:
        x    = sequences[batch_idx].to(DEVICE)
        mask = masks[batch_idx].to(DEVICE)

        visit_emb_clean, _ = model.encode(x, mask)
        x_feat_raw = x[:, :, 2:53]

        L_mae_val  = mae_loss(model, visit_emb_clean, x_feat_raw)
        L_egfr_val = egfr_loss(model, visit_emb_clean, x_feat_raw, mask)

        val_mae_list.append(L_mae_val.item())
        val_egfr_list.append(L_egfr_val.item())

    return {
        'val_egfr': np.mean(val_egfr_list),   # ← sole Early Stopping metric
        'val_mae':  np.mean(val_mae_list),
    }


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Data loading ────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    sequences = torch.from_numpy(
        np.load(SEQUENCES_PATH).astype(np.float32))
    masks = torch.from_numpy(np.load(MASKS_PATH))

    cohort = pd.read_csv(COHORT_PATH)
    assert 'split' in cohort.columns, \
        "Missing 'split' column (train/val) in the cohort file"

    train_idx = cohort[cohort['split'] == 'train'].index.tolist()
    val_idx   = cohort[cohort['split'] == 'val'].index.tolist()
    print(f"  Train: {len(train_idx)} patients, Val: {len(val_idx)} patients")
    print(f"  sequences: {sequences.shape}, D_INPUT used: {D_INPUT} (sex+race removed)")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_idx)),
        batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_idx)),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ── 2. Model & optimizer ───────────────────────────────────────
    print("\n[2/4] Initializing model...")
    model = Stage1Model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5)
    scaler = GradScaler()   # AMP gradient scaler

    # ── 3. Training loop ───────────────────────────────────────────
    print("\n[3/4] Starting training...")
    best_val_egfr  = float('inf')
    patience_count = 0
    log_rows       = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, train_loader, sequences, masks,
            optimizer, scaler)
        val_metrics   = validate(
            model, val_loader, sequences, masks)
        scheduler.step()

        # Early Stopping: monitor val_egfr only
        if val_metrics['val_egfr'] < best_val_egfr:
            best_val_egfr  = val_metrics['val_egfr']
            patience_count = 0
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, "stage1_best.pth")
            )
            flag = " ← best"
        else:
            patience_count += 1
            flag = f" (patience {patience_count}/{PATIENCE})"

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_metrics['train_loss']:.4f} "
            f"mae={train_metrics['train_mae']:.4f} "
            f"egfr={train_metrics['train_egfr']:.4f} "
            f"cont={train_metrics['train_contrast']:.4f} | "
            f"val_egfr={val_metrics['val_egfr']:.4f} "
            f"val_mae={val_metrics['val_mae']:.4f}"
            f"{flag}"
        )

        log_rows.append({
            'epoch':          epoch,
            'train_loss':     train_metrics['train_loss'],
            'train_mae':      train_metrics['train_mae'],
            'train_egfr':     train_metrics['train_egfr'],
            'train_contrast': train_metrics['train_contrast'],
            'train_smooth':   train_metrics['train_smooth'],
            'val_egfr':       val_metrics['val_egfr'],
            'val_mae':        val_metrics['val_mae'],
        })

        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── 4. Save training log ────────────────────────────────────────
    print("\n[4/4] Saving training log...")
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(OUTPUT_DIR, "stage1_train_log.csv"), index=False)

    print(f"\n✅ Training complete")
    print(f"   best val_egfr_loss = {best_val_egfr:.6f}")
    print(f"   Weights saved: stage1_best.pth")
    print(f"   Log saved: stage1_train_log.csv")
    print(f"\n→ Next step: run training/02_train_stage2_gcn_fusion.py (loads stage1_best.pth)")


if __name__ == "__main__":
    main()

