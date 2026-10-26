# Trajecta

Trajecta is a two-stage self-supervised framework that learns patient representations of chronic kidney disease (CKD) progression from longitudinal EHR. **Stage 1** pre-trains a Mamba encoder on each patient's visit sequence. **Stage 2** freezes the encoder and adds a GCN over a patient similarity graph, with gated fusion of the temporal and relational embeddings.

## Installation

Requires Linux, Python 3.10 and an NVIDIA GPU with CUDA 12 (tested on a single NVIDIA L4).

```bash
conda create -n trajecta python=3.10 -y
conda activate trajecta
pip install -r requirements.txt
```

## Quick start

Generate synthetic data in the model's input format and run training and inference:

```bash
python demo/make_synthetic_data.py
python demo/run_demo.py --epochs 3
```

Results are written to `demo/demo_output/`.

## Input data

| File | Format |
|---|---|
| `sequences.npy` | `(N, 500, 58)` float32. Each patient's visits (most recent 500), zero-padded |
| `masks.npy` | `(N, 500)` bool. `True` = real visit |
| `cohort.csv` | Columns `ID`, `split` (`train` / `val`), in the same row order as `sequences.npy` |
| `graph_edges.csv` | Columns `patient_i`, `patient_j`, `weight`: patient similarity graph |

The 58 features of each visit:

| Index | Features |
|---|---|
| 0–1 | Time since CKD diagnosis, interval since previous visit |
| 2–47 | 19 labs + 4 vitals as (value, availability) pairs. eGFR value is index 2 |
| 48–50 | Diabetes, hypertension, cardiovascular disease |
| 51 | CKD stage |
| 52–57 | Age, sex, race (4 one-hot) |

Continuous values are robust-scaled. Missing values are set to 0, with availability = 0. Sex and race are not fed to the encoder; they are used only to build the graph.

## Training

Set the file paths at the top of each script, then run:

```bash
python training/01_train_stage1_mamba.py      # -> stage1_best.pth
python training/02_train_stage2_gcn_fusion.py  # -> stage2_best.pth, patient_embedding.npy (N x 128)
```

Each stage takes about 2 h for N = 8,192 on one L4 GPU. Key settings are 128-dim embeddings, 4 Mamba blocks and 2 GCN layers. Training uses batch size 128, Adam (lr 1e-3) and up to 200 epochs with early stopping. The loss weights are reconstruction 0.6, next-visit eGFR 0.3, contrastive 0.1 and temporal smoothing 0.5.

## Inference

Encode a new cohort with the trained model and assign each patient to a trajectory segment, by *k*-NN against the training patients:

```bash
python inference/assign_validation_segments.py  # -> val_seg_assignment.csv, val_mamba_embedding.npy
```

This step needs the new cohort's `sequences.npy` / `masks.npy` / cohort CSV and the training patients' segment labels (an `.h5ad` file with `obs['seg']`).

## Citation

Citation information will be added upon publication.

## Contact

- Code questions: open a [GitHub issue](https://github.com/QSong-github/Trajecta/issues) or contact Sipei Gu (gusipei@ufl.edu)
- Corresponding author: Qianqian Song, PhD (song1081@purdue.edu)

## License

[MIT](LICENSE)
