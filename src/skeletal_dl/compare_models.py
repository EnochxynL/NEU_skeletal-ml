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

# -- Config -------------------------------------------------------------------

VAL_DATA = "./data/neu/val_data_joint.npy"
VAL_LABEL = "./data/neu/val_label.pkl"
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


def find_latest_ckpt(key):
    """Find the latest checkpoint for a given run key."""
    pattern = f"runs/neu_{key}_joint-*.pt"
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
    from skeletal_dl.model.stgcn import Model

    model = Model(
        num_class=60, num_point=25, num_person=2,
        graph="skeletal_dl.graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
    )
    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, 60


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
    print(f"Test set: {len(dataset)} samples, {len(set(dataset.labels))} classes\n")

    results = []

    # 1. Pre-trained ST-GCN (zero-shot)
    print(f"{'=' * 60}")
    print("[PT] ST-GCN (NTU-60 X-Sub pre-trained)")
    if os.path.exists(PRETRAINED_CKPT):
        model, num_class = load_pretrained_stgcn()
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        top1, top5, _, _ = evaluate(model, num_class, loader, device)
        print(f"  Params: {n_params:.2f}M  |  Top-1: {top1 * 100:.2f}%  |  Top-5: {top5 * 100:.2f}%")
        results.append({
            "name": "ST-GCN (pre-trained NTU-60)",
            "top1": top1, "top5": top5, "params_m": n_params,
        })
    else:
        print("  SKIP — checkpoint not found")
    print()

    # 2. Trained models (auto-discover checkpoints)
    for entry in MODEL_ENTRIES:
        print(f"{'=' * 60}")
        print(f"[{entry['key']}] {entry['name']}")

        ckpt_path = find_latest_ckpt(entry["key"])
        if ckpt_path is None:
            print(f"  SKIP — no checkpoint found (runs/neu_{entry['key']}_joint-*.pt)")
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
