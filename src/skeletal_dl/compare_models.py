"""
Batch evaluation: compare multiple trained models + pre-trained ST-GCN
on the NEU 10-class test set. Generates a comparison table and bar chart.

Usage:
    uv run python src/skeletal_dl/compare_models.py
"""

import os
import pickle
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

# Models to evaluate: (name, model_cls, checkpoint, num_class, model_args)
# Add entries here after training completes.
MODEL_ENTRIES = [
    {
        "name": "ST-GCN (pre-trained NTU-60)",
        "type": "pretrained",
        "checkpoint": PRETRAINED_CKPT,
    },
    {
        "name": "AGCN (scratch)",
        "type": "trained",
        "model": "skeletal_dl.model.agcn.Model",
        "checkpoint": None,  # fill after training, e.g. "runs/neu_agcn_joint-39-2730.pt"
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
        },
    },
    {
        "name": "ST-GCN (scratch)",
        "type": "trained",
        "model": "skeletal_dl.model.stgcn.Model",
        "checkpoint": None,  # fill after training
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
        },
    },
    {
        "name": "ST-GIN (scratch)",
        "type": "trained",
        "model": "skeletal_dl.model.stgin.Model",
        "checkpoint": None,  # fill after training
        "num_class": 10,
        "model_args": {
            "graph": "skeletal_dl.graph.ntu_rgb_d.Graph",
            "graph_args": {"labeling_mode": "spatial"},
        },
    },
]


def _import_class(name):
    import importlib
    module_path, class_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


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


def load_trained_model(entry):
    ModelCls = _import_class(entry["model"])
    model = ModelCls(num_class=entry["num_class"], num_point=25, num_person=2,
                     **entry["model_args"])
    weights = torch.load(entry["checkpoint"], map_location="cpu")
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

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars1 = ax.bar(x - width / 2, top1s, width, label="Top-1 Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, top5s, width, label="Top-5 Accuracy", color="coral")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison on NEU 10-class Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, top1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, top5s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SkeletonDataset(VAL_DATA, VAL_LABEL)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test set: {len(dataset)} samples, {len(set(dataset.labels))} classes\n")

    results = []
    for entry in MODEL_ENTRIES:
        print(f"{'=' * 60}")
        print(f"Model: {entry['name']}")

        if entry["type"] == "pretrained":
            model, num_class = load_pretrained_stgcn()
        elif entry["checkpoint"] is None:
            print("  -> SKIPPED (no checkpoint set)\n")
            continue
        else:
            if not os.path.exists(entry["checkpoint"]):
                print(f"  -> SKIPPED (checkpoint not found: {entry['checkpoint']})\n")
                continue
            model, num_class = load_trained_model(entry)

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        top1, top5, preds, labels = evaluate(model, num_class, loader, device)

        print(f"  Parameters: {n_params:.2f}M")
        print(f"  Top-1: {top1 * 100:.2f}%")
        print(f"  Top-5: {top5 * 100:.2f}%")

        results.append({
            "name": entry["name"],
            "top1": top1,
            "top5": top5,
            "params_m": n_params,
        })
        print()

    if not results:
        print("No models evaluated. Set checkpoint paths in MODEL_ENTRIES.")
        return

    # Table
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'Params':<8} {'Top-1':<10} {'Top-5':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['params_m']:.2f}M   {r['top1']*100:.2f}%      {r['top5']*100:.2f}%")
    print(f"{'=' * 70}")

    # Save text report
    txt_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")
    lines = [f"{'=' * 55}",
             "Model Comparison on NEU 10-class Test Set",
             f"{'=' * 55}", ""]
    for r in results:
        lines.append(f"  {r['name']}")
        lines.append(f"    Params: {r['params_m']:.2f}M")
        lines.append(f"    Top-1:  {r['top1']*100:.2f}%")
        lines.append(f"    Top-5:  {r['top5']*100:.2f}%")
        lines.append("")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {txt_path}")

    # Bar chart
    chart_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plot_comparison(results, chart_path)


if __name__ == "__main__":
    main()
