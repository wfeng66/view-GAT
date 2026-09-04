# View-GAT

View-based Graph Attention Network for multi-view 3D shape classification.

Each shape is represented by a set of rendered viewpoint images. View-GAT treats those views as nodes of a graph, aggregates them with **Graph Attention (GAT)** (or GCN), and hierarchically keeps the most informative views.

A related manuscript is in preparation and will be submitted soon.

## Model

Training has two stages.

**Stage 1 — single-view classifier (SVCNN).** An ImageNet-pretrained CNN is trained on individual views. The backbone later becomes the per-view feature extractor.

**Stage 2 — hierarchical view graph.** All views of a shape are encoded, then processed as a fully connected graph whose nodes carry view features and whose coordinates are the camera positions on a sphere (dodecahedron for 20 views, a 12-view layout, or Fibonacci sampling otherwise).

At each level the model:

1. Runs **Global GAT** or **Global GCN** over the current views (`--graph_net gat|gcn`).
2. Scores views with a **saliency head** (optionally mixed with GAT attention via `--att_lambda`).
3. **Selects top-k** views for the next, coarser level (hard or differentiable `--diff_topk`).
4. **Attention-pools** the current level into a single descriptor.

Pooled descriptors from all levels are concatenated and fed to an MLP classifier. Node count halves between levels when `--num_level` ≤ 3 (gentler 3/4 reduction for deeper stacks). Optional edge features encode camera geometry: 6-D `[v_i, v_j]` or 10-D `[v_i, v_j, v_i-v_j, |v_i-v_j|]`.

Backbones: ResNet, DenseNet, VGG, AlexNet.

## Installation

A CUDA GPU is required. Python 3.8+ is recommended (tested around 3.10).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# Install a CUDA build of PyTorch from https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

`torch-geometric` must match your PyTorch / CUDA versions. See the [PyG install notes](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if GAT/GCN layers fail to import.

## Data

Place already-rendered view images here:

```text
data/<dataset>_views/<class_name>/<train|test>/<model_id>_<view>.png
```

| Dataset      | Train path                        | Classes |
|--------------|-----------------------------------|---------|
| ModelNet40   | `data/modelnet40_views/*/train`   | 40      |
| ScanObjectNN | `data/scanobjectnn_views/*/train` | 15      |
| Colombia     | `data/colombia_views/*/train`     | 6       |
| RoofN3D      | `data/roofn3d_views/*/train`      | 2       |

Each shape should have **20** PNG views on disk (zero-based suffixes, e.g. `chair_000123_000.png` … `_019.png`). `--num_views` then subsamples that set evenly (for example 12 or 20).

## Training

```bash
python train.py \
  --name view-gat \
  --dataset modelnet40 \
  -num_views 20 \
  --cnn_name resnet18 \
  --graph_net gat \
  --num_level 3 \
  --stage1_epochs 25 \
  --stage2_epochs 20 \
  -weight_decay 0.001
```

ScanObjectNN (paths are set from `--dataset`):

```bash
python train.py \
  --name view-gat-sonn \
  --dataset scanobjectnn \
  -num_views 12 \
  -bs 12 \
  -lr 0.005 \
  --cnn_name resnet18 \
  --num_level 1 \
  --freeze_epochs 2
```

Checkpoints go to `<name>_stage_1/` and `<name>_stage_2/` (`*_best.pth`). Metrics for each run are appended to `exp_result.csv`.

```bash
tensorboard --logdir <name>_stage_2
```

### Main `train.py` arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `modelnet40` | `modelnet40`, `scanobjectnn`, `colombia`, `roofn3d` |
| `--name` | `view-gat` | Experiment name (log / checkpoint folder prefix) |
| `-bs` | `20` | Stage-2 batch size (number of shapes) |
| `-lr` | `1e-3` | Stage-2 learning rate |
| `-cnn_name` | `resnet18` | `resnet18/34/50/101/152`, `densenet121/161/169/201`, `vgg11`, `vgg16`, `alexnet` |
| `-num_views` | `20` | Views used at train time (subsampled from 20 on disk) |
| `--graph_net` | `gat` | `gat` or `gcn` |
| `--num_level` | `3` | Hierarchical graph levels |
| `--n_attn_heads` | `8` | Heads per level, e.g. `8` or `16,8,2` |
| `--edge_dim` | `None` | Omit, `6` (`[vi, vj]`), or `10` (`[vi, vj, vi-vj, \|vi-vj\|]`) |
| `--att_lambda` | `0.3` | Mix of GAT attention into saliency scores |
| `--diff_topk` | off | Differentiable top-k view selection |
| `--freeze_epochs` | `5` | Stage-1 epochs with a frozen backbone (ImageNet init) |
| `--no_pretraining` | off | Train the backbone from scratch |
| `--data_ver` / `--remark` | none | Tags stored in `exp_result.csv` |

## Repository layout

```text
train.py       # two-stage training
model/         # SVCNN + hierarchical View-GAT
tools/         # datasets, trainer, GAT/GCN layers
data/          # rendered views (not committed)
```
