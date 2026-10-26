"""
Step2: Stage 2 fine-tuning -- Mamba (fully frozen) + GCN + Gate Fusion

Goal:
  On top of the Stage1-pretrained Mamba, introduce a patient similarity graph (GCN).
  Mamba is fully frozen; only GCN / Gate Fusion / decoder heads / projection head are trained.

Model architecture:
  [inherited from Stage1, frozen] Input Encoding → Mamba SSM × 4 → Masked Mean Pool
  [new] Patient GCN × 2 → Gate Fusion σ
  → [main tasks] MAE reconstruction head / eGFR prediction head (applied to visit_emb_final after gate fusion)
  → [contrastive task] two independent DropEdge edge subsets → GCN → projection head → NT-Xent

Key design choices:
  - Mamba fully frozen: visit_emb / pat_emb are pre-cached before training (computed once)
  - At the start of each epoch: run the frozen cache through the GCN once to get pat_gcn_full for this epoch's gate fusion
  - Contrastive learning: DropEdge p=0.15, two independent edge subsets → different GCN outputs → NT-Xent
  - GateFusion initialization: bias=2.0 (sigmoid≈0.88), cold start biased toward the Mamba side
  - Transductive: val patients are also in the graph; the split only affects which samples contribute to the loss
  - Early Stopping: monitors val_egfr_loss only

Input files:
  - stage1_best.pth                    Stage1 pretrained weights
  - sequences.npy                      (8192, 500, 58)
  - masks.npy                          (8192, 500)
  - graph_edges.csv                    edges of the patient similarity graph (DTW + demographics)
  - cohort_8192_split.csv              contains split column (train / val)

Output files:
  - stage2_best.pth                    model weights with the best validation eGFR loss
  - patient_embedding.npy              (8192, 128) final patient embedding (GCN output)
  - visit_embedding.npy                (total real visits, 128) visit-level embedding (after Gate Fusion)
  - visit_index.csv                    (patient_id, visit_no) for each row of visit_embedding
  - stage2_train_log.csv               training log
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import GCNConv
from mamba_ssm import Mamba

# ============================================================
# Path configuration
# ============================================================
DATA_DIR        = "./8192_race_fixed/mamba_pretrain_results/3.0noslope/"
OUTPUT_DIR      = "./8192_race_fixed/finetune_results/3.0noslope/"
STAGE1_PTH      = "./8192_race_fixed/mamba_pretrain_results/3.0noslope/stage1_best.pth"
COHORT_PATH     = "./8192_race_fixed/cohort_8192_split.csv"
SEQUENCES_PATH  = "./8192_race_fixed/sequences.npy"
MASKS_PATH      = "./8192_race_fixed/masks.npy"
EDGES_PATH      = "./8192_race_fixed/graph_DTW_time_edges.csv"

# ============================================================
# Hyperparameters
# ============================================================
# Drop sex(idx53) + race_01~04(idx54~57), keep age(idx52)
D_INPUT      = 51
D_MODEL      = 128
D_PROJ       = 64
D_MAE_OUTPUT = 23
T_MAX        = 500

# x_feat = x[:,:,2:53]; EGFR_val (original col2) is idx=0 in x_feat
EGFR_VAL_IDX = 0

N_PATIENTS   = 8192
BATCH_SIZE   = 128
EPOCHS       = 200
TEMPERATURE  = 0.07
DROP_EDGE_P  = 0.15   # DropEdge random drop ratio per pass

# Learning rates (Mamba fully frozen, not in the optimizer)
LR_GCN       = 1e-3
LR_GATE      = 1e-3
LR_DECODER   = 1e-4
LR_PROJ      = 1e-3

# Loss weights
W_MAE      = 0.6
W_EGFR     = 0.3
W_CONTRAST = 0.1
W_SMOOTH   = 0.5

PATIENCE   = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# Shared model components (same as Step1)
# ============================================================

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


# ============================================================
# Full Stage 2 model
# ============================================================

class Stage2Model(nn.Module):
    """
    Stage 2 model: Mamba (frozen) + GCN × 2 + Gate Fusion

    encode()       → visit_emb, pat_emb     pure Mamba outputs (cached once before training)
    gcn_forward()  → (8192, D_MODEL)        full-graph GCN propagation
    gate_fusion()  → visit_emb_final        gated fusion of Mamba + GCN
    """
    def __init__(self, d_input=D_INPUT, d_model=D_MODEL,
                 d_proj=D_PROJ, d_mae=D_MAE_OUTPUT, n_mamba_layers=4):
        super().__init__()

        # ── Mamba backbone (loaded from Stage1, fully frozen) ──
        self.input_proj   = nn.Linear(d_input, d_model)
        self.input_norm   = nn.LayerNorm(d_model)
        self.time_encoder = SinusoidalTimeEncoder(d_model)
        self.mamba_blocks = nn.ModuleList(
            [MambaBlock(d_model) for _ in range(n_mamba_layers)]
        )

        # ── Decoder heads (loaded from Stage1, medium learning rate) ──
        self.mae_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_mae),
        )
        self.egfr_head = nn.Linear(d_model, 1)

        # ── GCN × 2 (random init, normal learning rate) ──
        self.gcn1      = GCNConv(d_model, d_model)
        self.gcn1_norm = nn.LayerNorm(d_model)
        self.gcn2      = GCNConv(d_model, d_model)
        self.gcn2_norm = nn.LayerNorm(d_model)

        # ── Gate Fusion (random init, normal learning rate) ──
        self.gate_linear = nn.Linear(d_model * 2, d_model)
        # Cold start: gain=0.01 suppresses the effect of random GCN weights; bias=2.0 gives initial gate≈0.88 (Mamba-biased)
        nn.init.xavier_uniform_(self.gate_linear.weight, gain=0.01)
        nn.init.constant_(self.gate_linear.bias, 2.0)

        # ── Contrastive projection head (random init, on GCN output) ──
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

    def encode(self, x, mask):
        """
        x:    (B, T, 58)
        mask: (B, T)

        Returns:
          visit_emb: (B, T, D_MODEL)
          pat_emb:   (B, D_MODEL)  masked mean pool
        """
        x_feat = x[:, :, 2:53]    # (B, T, 51)
        x_time = x[:, :, 0:2]     # (B, T, 2)

        h = self.input_norm(
            self.input_proj(x_feat) + self.time_encoder(x_time)
        )
        for block in self.mamba_blocks:
            h = block(h)

        visit_emb = h
        mask_f    = mask.unsqueeze(-1).float()
        pat_emb   = (visit_emb * mask_f).sum(dim=1) \
                    / mask_f.sum(dim=1).clamp(min=1e-8)

        return visit_emb, pat_emb

    def gcn_forward(self, all_pat_emb, edge_index, edge_weight):
        """
        all_pat_emb:  (N, D_MODEL)  full-graph patient embeddings (8192 or a subset)
        edge_index:   (2, E)
        edge_weight:  (E,)

        Returns: (N, D_MODEL)
        """
        h = F.relu(self.gcn1_norm(
            self.gcn1(all_pat_emb, edge_index, edge_weight)
        ))
        h = self.gcn2_norm(
            self.gcn2(h, edge_index, edge_weight)
        )
        h = h + all_pat_emb   # residual: prevents over-smoothing
        return h

    def gate_fusion(self, visit_emb, pat_gcn_batch):
        """
        visit_emb:     (B, T, D_MODEL)  Mamba temporal output
        pat_gcn_batch: (B, D_MODEL)     GCN output (patient level)

        Returns: (B, T, D_MODEL)
        """
        gcn_exp = pat_gcn_batch.unsqueeze(1).expand(-1, visit_emb.size(1), -1)
        gate = torch.sigmoid(
            self.gate_linear(torch.cat([visit_emb, gcn_exp], dim=-1))
        )
        return gate * visit_emb + (1.0 - gate) * gcn_exp


# ============================================================
# DropEdge (graph augmentation for contrastive learning)
# ============================================================

def drop_edge(edge_index, edge_weight, p=DROP_EDGE_P):
    """Randomly drop a fraction p of edges and return the retained subset"""
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) > p
    return edge_index[:, keep], edge_weight[keep]


# ============================================================
# Full-graph embedding pre-cache (called once before training; fixed since Mamba is frozen)
# ============================================================

@torch.no_grad()
def build_full_embedding_cache(model, sequences, masks):
    """
    Run a single encode over all 8192 patients with the frozen Mamba.

    Returns:
      visit_emb_cache: (8192, T_MAX, D_MODEL)  CPU tensor (2GB, kept off VRAM)
      pat_emb_cache:   (8192, D_MODEL)          CPU tensor (4MB, moved to GPU by main)
    """
    model.eval()
    visit_list = []
    pat_list   = []

    for i in range(0, N_PATIENTS, 256):
        x_b    = sequences[i:i+256].to(DEVICE)
        mask_b = masks[i:i+256].to(DEVICE)
        with autocast():
            visit_emb, pat_emb = model.encode(x_b, mask_b)
        visit_list.append(visit_emb.cpu())
        pat_list.append(pat_emb.cpu())

    model.train()
    visit_emb_cache = torch.cat(visit_list, dim=0)   # (8192, T_MAX, D_MODEL), CPU
    pat_emb_cache   = torch.cat(pat_list,   dim=0)   # (8192, D_MODEL), CPU

    print(f"  visit_emb_cache: {visit_emb_cache.shape}  "
          f"({visit_emb_cache.nbytes / 1024**3:.2f} GB, CPU)")
    print(f"  pat_emb_cache:   {pat_emb_cache.shape}  (CPU → moved to GPU)")

    return visit_emb_cache, pat_emb_cache


# ============================================================
# Loss functions
# ============================================================

def mae_loss(model, visit_emb, x_feat_raw):
    pred          = model.mae_decoder(visit_emb)
    val_indices   = list(range(0, 46, 2))[:23]
    avail_indices = list(range(1, 47, 2))[:23]
    target        = x_feat_raw[:, :, val_indices]
    avail         = x_feat_raw[:, :, avail_indices]
    valid_mask    = avail > 0.5
    if valid_mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[valid_mask], target[valid_mask])


def egfr_loss(model, visit_emb, x_feat_raw, mask):
    pred      = model.egfr_head(visit_emb[:, :-1, :]).squeeze(-1)
    target    = x_feat_raw[:, 1:, EGFR_VAL_IDX]
    avail     = x_feat_raw[:, 1:, EGFR_VAL_IDX + 1]
    next_mask = mask[:, 1:] & (avail > 0.5)
    if next_mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[next_mask], target[next_mask])


def smooth_loss(visit_emb, mask, x_time_raw):
    e1       = visit_emb[:, :-1, :]
    e2       = visit_emb[:, 1:,  :]
    cos_sim  = F.cosine_similarity(e1, e2, dim=-1)
    # interval is in raw days; decay weight with a 90-day scale
    interval = x_time_raw[:, 1:, 1]
    weight   = torch.exp(-interval.abs() / 90.0)
    valid    = mask[:, :-1] & mask[:, 1:]
    if valid.sum() == 0:
        return cos_sim.sum() * 0.0
    return ((1.0 - cos_sim) * weight)[valid].mean()


def contrastive_loss_stage2(model, pat_emb_cache_gpu, batch_idx_t,
                             edge_index, edge_weight):
    """
    DropEdge NT-Xent (Stage 2):
      two independent DropEdge passes → each through GCN → take batch → project → L2 normalize → NT-Xent

    pat_emb_cache_gpu: (8192, D_MODEL) on GPU, frozen Mamba cache
    batch_idx_t:       (B,) LongTensor, global node indices of the current batch
    """
    B = batch_idx_t.shape[0]

    ei_v1, ew_v1 = drop_edge(edge_index, edge_weight)
    ei_v2, ew_v2 = drop_edge(edge_index, edge_weight)

    gcn_v1 = model.gcn_forward(pat_emb_cache_gpu, ei_v1, ew_v1)  # (8192, D)
    gcn_v2 = model.gcn_forward(pat_emb_cache_gpu, ei_v2, ew_v2)  # (8192, D)

    z1 = F.normalize(model.proj_head(gcn_v1[batch_idx_t]), dim=-1)  # (B, D_PROJ)
    z2 = F.normalize(model.proj_head(gcn_v2[batch_idx_t]), dim=-1)  # (B, D_PROJ)

    z   = torch.cat([z1, z2], dim=0)         # (2B, D_PROJ)
    sim = z @ z.T / TEMPERATURE              # (2B, 2B)
    sim.fill_diagonal_(float('-inf'))

    labels = torch.cat([
        torch.arange(B, 2 * B, device=DEVICE),
        torch.arange(0, B,     device=DEVICE),
    ])
    return F.cross_entropy(sim, labels)


# ============================================================
# Training & validation
# ============================================================

def train_one_epoch(model, loader, sequences, masks,
                    visit_emb_cache, pat_emb_cache_gpu,
                    optimizer, scaler,
                    edge_index, edge_weight):
    model.train()
    total = dict(loss=0, mae=0, egfr=0, contrast=0, smooth=0)
    n_batches = 0

    for (batch_idx_t,) in loader:
        # O(1) lookup: no Mamba forward pass (frozen and cached)
        visit_emb_clean = visit_emb_cache[batch_idx_t].to(DEVICE)  # (B, T, D)

        x    = sequences[batch_idx_t].to(DEVICE)
        mask = masks[batch_idx_t].to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            # Full-graph GCN (with gradients; eGFR/MAE errors backpropagate directly into GCN parameters)
            pat_gcn_full  = model.gcn_forward(pat_emb_cache_gpu, edge_index, edge_weight)
            pat_gcn_batch = pat_gcn_full[batch_idx_t]

            # ── Main tasks: Gate Fusion → decoder heads ─────────────────────
            visit_emb_final = model.gate_fusion(visit_emb_clean, pat_gcn_batch)

            x_feat_raw = x[:, :, 2:53]
            x_time_raw = x[:, :, 0:2]

            L_mae    = mae_loss(model, visit_emb_final, x_feat_raw)
            L_egfr   = egfr_loss(model, visit_emb_final, x_feat_raw, mask)
            L_smooth = smooth_loss(visit_emb_final, mask, x_time_raw)

            # ── Contrastive task: DropEdge → GCN → NT-Xent ───────────────
            L_contrast = contrastive_loss_stage2(
                model, pat_emb_cache_gpu, batch_idx_t,
                edge_index, edge_weight)

            loss = (W_MAE      * L_mae     +
                    W_EGFR     * L_egfr    +
                    W_CONTRAST * L_contrast +
                    W_SMOOTH   * L_smooth)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total['loss']     += loss.item()
        total['mae']      += L_mae.item()
        total['egfr']     += L_egfr.item()
        total['contrast'] += L_contrast.item()
        total['smooth']   += L_smooth.item()
        n_batches         += 1

    return {f'train_{k}': v / n_batches for k, v in total.items()}


@torch.no_grad()
def validate(model, loader, sequences, masks,
             visit_emb_cache, pat_emb_cache_gpu,
             edge_index, edge_weight):
    model.eval()
    val_egfr_list = []
    val_mae_list  = []

    # Weights are fixed during validation, so the full graph is computed once
    with autocast():
        pat_gcn_full = model.gcn_forward(pat_emb_cache_gpu, edge_index, edge_weight)

    for (batch_idx_t,) in loader:
        visit_emb_clean = visit_emb_cache[batch_idx_t].to(DEVICE)

        x    = sequences[batch_idx_t].to(DEVICE)
        mask = masks[batch_idx_t].to(DEVICE)

        with autocast():
            pat_gcn_batch   = pat_gcn_full[batch_idx_t]
            visit_emb_final = model.gate_fusion(visit_emb_clean, pat_gcn_batch)
            x_feat_raw  = x[:, :, 2:53]
            L_mae_val   = mae_loss(model, visit_emb_final, x_feat_raw)
            L_egfr_val  = egfr_loss(model, visit_emb_final, x_feat_raw, mask)

        val_egfr_list.append(L_egfr_val.item())
        val_mae_list.append(L_mae_val.item())

    return {
        'val_egfr': np.mean(val_egfr_list),   # ← sole Early Stopping metric
        'val_mae':  np.mean(val_mae_list),
    }


# ============================================================
# Final embedding export
# ============================================================

@torch.no_grad()
def extract_and_save(model, visit_emb_cache, pat_emb_cache_gpu,
                     edge_index, edge_weight, cohort, masks):
    """
    Extract and save embeddings for all 8192 patients with the best model.
    Uses the pre-cache directly (Mamba frozen, outputs unchanged); only GCN + Gate Fusion are rerun.
    """
    print("\nExtracting final embeddings...")
    model.eval()

    # ── GCN propagation (full graph, full edge set) ──────────────────────────────
    with autocast():
        pat_emb_gcn = model.gcn_forward(
            pat_emb_cache_gpu, edge_index, edge_weight)   # (8192, D_MODEL)

    np.save(
        os.path.join(OUTPUT_DIR, "patient_embedding.npy"),
        pat_emb_gcn.cpu().float().numpy()
    )
    print(f"  ✅ patient_embedding.npy: {pat_emb_gcn.shape}")

    # ── Gate Fusion → visit_embedding ─────────────────────────
    visit_emb_final_list = []
    visit_index_rows     = []
    patient_ids          = cohort['ID'].tolist()

    for i in range(0, N_PATIENTS, 256):
        end    = min(i + 256, N_PATIENTS)
        v_emb  = visit_emb_cache[i:end].to(DEVICE)   # (B, T, D_MODEL)
        v_mask = masks[i:end]                          # (B, T) bool, CPU
        gcn_b  = pat_emb_gcn[i:end]                   # (B, D_MODEL)

        with autocast():
            fused = model.gate_fusion(v_emb, gcn_b)   # (B, T, D_MODEL)

        for b_idx in range(end - i):
            real_visits = v_mask[b_idx]                     # (T,) bool, CPU
            n_real      = real_visits.sum().item()
            global_idx  = i + b_idx
            pid         = patient_ids[global_idx]

            fused_real  = fused[b_idx].cpu()[real_visits]  # (n_real, D_MODEL), CPU
            visit_emb_final_list.append(fused_real.float())

            for visit_no in range(int(n_real)):
                visit_index_rows.append({
                    'patient_id': pid,
                    'visit_no':   visit_no + 1,
                    'global_idx': global_idx,
                })

    visit_emb_all = torch.cat(visit_emb_final_list, dim=0).numpy()
    np.save(os.path.join(OUTPUT_DIR, "visit_embedding.npy"), visit_emb_all)
    pd.DataFrame(visit_index_rows).to_csv(
        os.path.join(OUTPUT_DIR, "visit_index.csv"), index=False)

    print(f"  ✅ visit_embedding.npy:   {visit_emb_all.shape}")
    print(f"  ✅ visit_index.csv:       {len(visit_index_rows)} visit records")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Data loading ────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    sequences = torch.from_numpy(
        np.load(SEQUENCES_PATH).astype(np.float32))
    masks = torch.from_numpy(np.load(MASKS_PATH))

    cohort = pd.read_csv(COHORT_PATH)
    assert 'split' in cohort.columns, \
        "Missing 'split' column (train/val) in the cohort file"

    train_idx = cohort[cohort['split'] == 'train'].index.tolist()
    val_idx   = cohort[cohort['split'] == 'val'].index.tolist()
    print(f"  Train: {len(train_idx)} patients, Val: {len(val_idx)} patients")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_idx)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_idx)),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ── 2. Graph loading ──────────────────────────────────────────
    print("\n[2/5] Loading patient similarity graph...")
    edges_df   = pd.read_csv(EDGES_PATH)
    cohort_ids = cohort['ID'].tolist()
    id_to_idx  = {pid: i for i, pid in enumerate(cohort_ids)}

    src = [id_to_idx[pid] for pid in edges_df['patient_i']]
    dst = [id_to_idx[pid] for pid in edges_df['patient_j']]

    src_bi = src + dst
    dst_bi = dst + src
    wt_bi  = edges_df['weight'].tolist() * 2

    edge_index  = torch.tensor([src_bi, dst_bi], dtype=torch.long).to(DEVICE)
    edge_weight = torch.tensor(wt_bi, dtype=torch.float32).to(DEVICE)

    print(f"  Nodes: {N_PATIENTS}")
    print(f"  Directed edges: {edge_index.shape[1]} ({len(src)} undirected edges × 2)")

    # ── 3. Model initialization & loading Stage1 weights ──────────────────────
    print("\n[3/5] Initializing model and loading Stage1 weights...")
    model = Stage2Model().to(DEVICE)

    stage1_state = torch.load(STAGE1_PTH, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(stage1_state, strict=False)
    print(f"  Stage1 weights loaded")
    print(f"  New parameters (not loaded): {missing}")
    print(f"  Unexpected parameters (ignored): {unexpected}")

    # ── Freeze Mamba (not in optimizer; gradients are meaningless after caching) ──────────
    mamba_params = (list(model.input_proj.parameters())
                  + list(model.input_norm.parameters())
                  + list(model.time_encoder.parameters())
                  + list(model.mamba_blocks.parameters()))
    for p in mamba_params:
        p.requires_grad = False
    frozen_n = sum(p.numel() for p in mamba_params)
    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Mamba frozen: {frozen_n:,} parameters")
    print(f"  Trainable parameters: {trainable_n:,}")

    # ── 4. Pre-cache Mamba embeddings (computed once) ──────────────────
    print("\n[4/5] Pre-caching Mamba embeddings (fixed after freezing)...")
    visit_emb_cache, pat_emb_cache = build_full_embedding_cache(
        model, sequences, masks)
    pat_emb_cache_gpu = pat_emb_cache.to(DEVICE)   # (8192, D_MODEL), 4MB on GPU

    # ── Optimizer (trainable parameters only; Mamba excluded) ──────────
    decoder_params = (list(model.mae_decoder.parameters())
                    + list(model.egfr_head.parameters()))
    gcn_params     = (list(model.gcn1.parameters())
                    + list(model.gcn1_norm.parameters())
                    + list(model.gcn2.parameters())
                    + list(model.gcn2_norm.parameters()))
    gate_params    = list(model.gate_linear.parameters())
    proj_params    = list(model.proj_head.parameters())

    optimizer = torch.optim.Adam([
        {'params': decoder_params, 'lr': LR_DECODER, 'name': 'decoder'},
        {'params': gcn_params,     'lr': LR_GCN,     'name': 'gcn'},
        {'params': gate_params,    'lr': LR_GATE,     'name': 'gate'},
        {'params': proj_params,    'lr': LR_PROJ,     'name': 'proj'},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=1e-6)
    scaler = GradScaler()

    # ── 5. Training loop ───────────────────────────────────────────
    print("\n[5/5] Starting training...")
    best_val_egfr  = float('inf')
    patience_count = 0
    log_rows       = []

    for epoch in range(1, EPOCHS + 1):

        train_metrics = train_one_epoch(
            model, train_loader, sequences, masks,
            visit_emb_cache, pat_emb_cache_gpu,
            optimizer, scaler,
            edge_index, edge_weight)

        val_metrics = validate(
            model, val_loader, sequences, masks,
            visit_emb_cache, pat_emb_cache_gpu,
            edge_index, edge_weight)

        scheduler.step()

        if val_metrics['val_egfr'] < best_val_egfr:
            best_val_egfr  = val_metrics['val_egfr']
            patience_count = 0
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, "stage2_best.pth"))
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

    # ── Extract and save final embeddings ────────────────────────────────
    print("\nLoading best model and extracting embeddings...")
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "stage2_best.pth"),
                   map_location=DEVICE))

    extract_and_save(model, visit_emb_cache, pat_emb_cache_gpu,
                     edge_index, edge_weight, cohort, masks)

    pd.DataFrame(log_rows).to_csv(
        os.path.join(OUTPUT_DIR, "stage2_train_log.csv"), index=False)

    print(f"\n✅ Stage 2 complete")
    print(f"   best val_egfr_loss = {best_val_egfr:.6f}")
    print(f"   → patient_embedding.npy can be used directly for UMAP visualization")


if __name__ == "__main__":
    main()
