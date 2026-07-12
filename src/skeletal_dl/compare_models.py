"""
Batch evaluation: compare multiple trained models + pre-trained ST-GCN
on the NEU 10-class test set. Generates a comparison table and bar chart.

Auto-discovers checkpoints from runs/ directory.
Specify None as checkpoint to auto-find the latest.

Usage:
    uv run python src/skeletal_dl/compare_models.py
"""

import os
import pickle
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Enable Chinese font display on Windows
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -- Config -------------------------------------------------------------------

VAL_DATA = "./data/neu/val_data_joint.npy"
VAL_LABEL = "./data/neu/val_label.pkl"
PT_DATA = "D:/ACTIVE/NEUAI/mmskeleton/data/NEU/test_data.npy"
PT_LABEL = "D:/ACTIVE/NEUAI/mmskeleton/data/NEU/test_label.pkl"
PRETRAINED_CKPT = "D:/ACTIVE/NEUAI/mmskeleton/checkpoints/st_gcn.ntu-xsub-300b57d4.pth"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 32

# Model entries: (display_name, run_key, model_path, num_class, model_args)
# run_key is used to auto-find checkpoint: runs/neu_{key}_joint-*.pt
MODEL_ENTRIES = [
    {
        "name": "ST-GCN (scratch baseline)",
        "key": "stgcn",
        "model": "skeletal_dl.model.stgcn.Model",
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
        },
    },
    {
        "name": "ST-GCN (+augmentation)",
        "key": "stgcn_aug",
        "ckpt_pattern": "runs/neu_stgcn_joint_aug-*.pt",
        "model": "skeletal_dl.model.stgcn.Model",
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
        },
    },
    {
        "name": "ST-GCN (+LR+dropout)",
        "key": "stgcn_lr",
        "ckpt_pattern": "runs/neu_stgcn_joint_lr-*.pt",
        "model": "skeletal_dl.model.stgcn.Model",
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
            "dropout": 0.5,
        },
    },
    {
        "name": "AGCN (scratch baseline)",
        "key": "agcn",
        "model": "skeletal_dl.model.agcn.Model",
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
        },
    },
    {
        "name": "AGCN (+dropout)",
        "key": "agcn_dropout",
        "ckpt_pattern": "runs/neu_agcn_joint_dropout-*.pt",
        "model": "skeletal_dl.model.agcn_dropout.Model",
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
            "dropout": 0.5,
        },
    },
    {
        "name": "ST-GIN (+dropout)",
        "key": "stgin",
        "model": "skeletal_dl.model.stgin.Model",
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
            "dropout": 0.5,
        },
    },
    {
        "name": "ResNet+VirtualRadar",
        "key": "resnet",
        "model": "skeletal_dl.model.resnet.Model",
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
            "num_filters": 64,
            "image_size": 256,
            "wavelength": 0.0005,
        },
    },
]


def _import_class(name):
    import importlib

    module_path, class_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def find_latest_ckpt(entry):
    """Find the latest checkpoint for a given model entry."""
    pattern = entry.get("ckpt_pattern", f"runs/neu_{entry['key']}_joint-*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


class SkeletonDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.array(np.load(data_path, mmap_mode="r"), dtype=np.float32)
        with open(label_path, "rb") as f:
            self.sample_names, self.labels = pickle.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


def load_pretrained_stgcn():
    """Build ST-GCN with the original paper's spatial graph partitioning.

    The mmskeleton pre-trained weights were trained with the original
    root/centripetal/centrifugal partitioning, not the 2s-AGCN inward/outward
    partitioning used elsewhere in this project.
    """
    from skeletal_dl.model.stgcn import Model

    model = Model(
        num_class=60, num_point=25, num_person=2,
        graph="skeletal_dl.graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
    )
    # Replace with original ST-GCN graph (matches pre-trained weights)
    ref_A = _build_pretrained_graph()
    model.register_buffer('A', torch.tensor(ref_A, dtype=torch.float32))

    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, 60


def _build_pretrained_graph():
    """Original ST-GCN spatial adjacency matrix (paper version).

    Root / centripetal / centrifugal partitioning with global normalization.
    The pre-trained edge_importance weights were trained on this graph.
    """
    num_node = 25
    self_link = [(i, i) for i in range(num_node)]
    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                      (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                      (11, 10), (12, 11), (13, 1), (14, 13),
                      (15, 14), (16, 15), (17, 1), (18, 17),
                      (19, 18), (20, 19), (22, 23), (23, 8),
                      (24, 25), (25, 12)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    edge = self_link + neighbor_link
    center = 21 - 1

    max_hop = 1
    A_mat = np.zeros((num_node, num_node))
    for i, j in edge:
        A_mat[j, i] = 1
        A_mat[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A_mat, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    adjacency = np.zeros((num_node, num_node))
    for hop in range(max_hop + 1):
        adjacency[hop_dis == hop] = 1
    Dl = np.sum(adjacency, axis=0)
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    normalize_adjacency = np.dot(adjacency, Dn)

    valid_hop = range(0, max_hop + 1)
    A = []
    for hop in valid_hop:
        a_root = np.zeros((num_node, num_node))
        a_close = np.zeros((num_node, num_node))
        a_further = np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if hop_dis[j, i] == hop:
                    if hop_dis[j, center] == hop_dis[i, center]:
                        a_root[j, i] = normalize_adjacency[j, i]
                    elif hop_dis[j, center] > hop_dis[i, center]:
                        a_close[j, i] = normalize_adjacency[j, i]
                    else:
                        a_further[j, i] = normalize_adjacency[j, i]
        if hop == 0:
            A.append(a_root)
        else:
            A.append(a_root + a_close)
            A.append(a_further)

    return np.stack(A)


def load_trained_model(entry, ckpt_path):
    ModelCls = _import_class(entry["model"])
    model = ModelCls(num_class=entry["num_class"], num_point=25, num_person=2,
                     **entry["model_args"])
    weights = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model, entry["num_class"]


def evaluate(model, num_class, loader, device):
    all_scores, all_labels = [], []
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            all_scores.append(output.cpu())
            all_labels.append(label.cpu())

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = scores.argmax(dim=1)

    top1 = (preds == labels).float().mean().item()
    top5 = topk_accuracy(scores, labels, k=5)
    return top1, top5, preds, labels


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit = [int(l in rank[i, -k:]) for i, l in enumerate(label)]
    return sum(hit) / len(hit)


def plot_comparison(results, save_path):
    names = [r["name"] for r in results]
    top1s = [r["top1"] * 100 for r in results]
    top5s = [r["top5"] * 100 for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.3), 5))
    bars1 = ax.bar(x - width / 2, top1s, width, label="Top-1", color="steelblue")
    bars2 = ax.bar(x + width / 2, top5s, width, label="Top-5", color="coral")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("模型对比 — NEU 10-class 测试集")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, top1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                fontweight="bold")
    for bar, val in zip(bars2, top5s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = SkeletonDataset(VAL_DATA, VAL_LABEL)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f"Test set (pre_normalized): {len(dataset)} samples, {len(set(dataset.labels))} classes")

    # Separate dataset for pre-trained model (raw coords, no pre_normalization)
    pt_dataset = SkeletonDataset(PT_DATA, PT_LABEL) if os.path.exists(PT_DATA) else None
    if pt_dataset:
        pt_loader = DataLoader(pt_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=True)
        print(f"PT test set (raw): {len(pt_dataset)} samples, {len(set(pt_dataset.labels))} classes")
    print()

    results = []

    # 1. Pre-trained ST-GCN (zero-shot, uses raw data without pre_normalization)
    print(f"{'=' * 60}")
    print("[PT] ST-GCN (NTU-60 X-Sub pre-trained)")
    if os.path.exists(PRETRAINED_CKPT) and pt_dataset is not None:
        model, num_class = load_pretrained_stgcn()
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        top1, top5, _, _ = evaluate(model, num_class, pt_loader, device)
        print(f"  Params: {n_params:.2f}M  |  Top-1: {top1 * 100:.2f}%  |  Top-5: {top5 * 100:.2f}%")
        results.append({
            "name": "ST-GCN (pre-trained NTU-60)",
            "top1": top1, "top5": top5, "params_m": n_params,
        })
    else:
        print("  SKIP — checkpoint or raw data not found")
    print()

    # 2. Trained models (auto-discover checkpoints)
    for entry in MODEL_ENTRIES:
        print(f"{'=' * 60}")
        print(f"[{entry['key']}] {entry['name']}")

        ckpt_path = find_latest_ckpt(entry)
        if ckpt_path is None:
            print(f"  SKIP — no checkpoint found")
            print()
            continue

        print(f"  checkpoint: {ckpt_path}")
        try:
            model, num_class = load_trained_model(entry, ckpt_path)
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            print()
            continue

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        top1, top5, _, _ = evaluate(model, num_class, loader, device)
        print(f"  Params: {n_params:.2f}M  |  Top-1: {top1 * 100:.2f}%  |  Top-5: {top5 * 100:.2f}%")
        results.append({
            "name": entry["name"],
            "top1": top1, "top5": top5, "params_m": n_params,
        })
        print()

    if not results:
        print("No models evaluated. Train models first, then re-run.")
        return

    # Summary table
    print(f"\n{'=' * 75}")
    print(f"{'Model':<35} {'Params':<10} {'Top-1':<12} {'Top-5':<12}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: -x["top1"]):
        print(f"{r['name']:<35} {r['params_m']:.2f}M     {r['top1']*100:>6.2f}%     {r['top5']*100:>6.2f}%")
    print(f"{'=' * 75}")

    # Save text report
    txt_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")
    lines = [
        f"{'=' * 55}",
        "Model Comparison — NEU 10-class Test Set",
        f"{'=' * 55}",
        "",
        f"{'Model':<35} {'Top-1':<10} {'Top-5':<10} {'Params':<10}",
        "-" * 65,
    ]
    for r in sorted(results, key=lambda x: -x["top1"]):
        lines.append(
            f"{r['name']:<35} {r['top1']*100:>6.2f}%   {r['top5']*100:>6.2f}%   {r['params_m']:.2f}M"
        )
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {txt_path}")

    # Bar chart (exclude pre-trained to keep scale meaningful, or include all)
    chart_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plot_comparison(results, chart_path)


if __name__ == "__main__":
    main()
