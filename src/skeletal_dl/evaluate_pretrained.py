"""
Evaluate pre-trained ST-GCN (NTU-60 X-Sub) on NEU 10-class test data.

The mmskeleton checkpoint was trained on NTU-60 (classes A001-A060).
NEU data uses classes A001-A010, which map directly to NTU-60 classes 0-9.

Usage:
    uv run python src/skeletal_dl/evaluate_pretrained.py eval
    uv run python src/skeletal_dl/evaluate_pretrained.py predict <file.skeleton>
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -- Paths --
CHECKPOINT_PATH = "D:/ACTIVE/NEUAI/mmskeleton/checkpoints/st_gcn.ntu-xsub-300b57d4.pth"
DATA_PATH = "D:/ACTIVE/NEUAI/mmskeleton/data/NEU/test_data.npy"
LABEL_PATH = "D:/ACTIVE/NEUAI/mmskeleton/data/NEU/test_label.pkl"
NUM_CLASS = 60  # NTU-60 pre-trained model
NUM_NEU_CLASSES = 10

ACTION_NAMES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop",
    "pick up", "throw", "sit down", "stand up", "clapping",
]


def build_model():
    """Build ST-GCN with the ORIGINAL ST-GCN spatial graph (paper version).

    The mmskeleton pre-trained checkpoint was trained with the original
    spatial configuration partitioning (root / centripetal / centrifugal),
    NOT the 2s-AGCN inward/outward partitioning used elsewhere in this project.
    We construct the correct adjacency matrix and inject it into the model.
    """
    from skeletal_dl.model.stgcn import Model

    model = Model(
        num_class=NUM_CLASS,
        num_point=25,
        num_person=2,
        graph="skeletal_dl.graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
    )

    # Replace adjacency matrix with the original ST-GCN spatial partitioning
    ref_A = _build_original_stgcn_graph()
    model.register_buffer('A', torch.tensor(ref_A, dtype=torch.float32))

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _build_original_stgcn_graph():
    """Construct the original ST-GCN spatial adjacency matrix.

    This matches the paper's partitioning (root / centripetal / centrifugal
    with global normalization), exactly as in the reference mmskeleton
    implementation.  The pre-trained weights were trained on this graph.
    """
    import numpy as np

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

    # Normalize full adjacency matrix
    adjacency = np.zeros((num_node, num_node))
    for hop in range(max_hop + 1):
        adjacency[hop_dis == hop] = 1
    Dl = np.sum(adjacency, axis=0)
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    normalize_adjacency = np.dot(adjacency, Dn)

    # Spatial partitioning: root / centripetal / centrifugal
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


def read_skeleton(file):
    with open(file, "r") as f:
        skeleton_sequence = {}
        skeleton_sequence["numFrame"] = int(f.readline())
        skeleton_sequence["frameInfo"] = []
        for t in range(skeleton_sequence["numFrame"]):
            frame_info = {}
            frame_info["numBody"] = int(f.readline())
            frame_info["bodyInfo"] = []
            for m in range(frame_info["numBody"]):
                body_info_key = [
                    "bodyID", "clipedEdges", "handLeftConfidence",
                    "handLeftState", "handRightConfidence", "handRightState",
                    "isResticted", "leanX", "leanY", "trackingState",
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info["numJoint"] = int(f.readline())
                body_info["jointInfo"] = []
                for v in range(body_info["numJoint"]):
                    joint_info_key = [
                        "x", "y", "z", "depthX", "depthY", "colorX", "colorY",
                        "orientationW", "orientationX", "orientationY",
                        "orientationZ", "trackingState",
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info["jointInfo"].append(joint_info)
                frame_info["bodyInfo"].append(body_info)
            skeleton_sequence["frameInfo"].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25, max_frame=300):
    seq_info = read_skeleton(file)
    data = np.zeros((3, max_frame, num_joint, max_body), dtype=np.float32)
    for n, f in enumerate(seq_info["frameInfo"]):
        for m, b in enumerate(f["bodyInfo"]):
            for j, v in enumerate(b["jointInfo"]):
                if m < max_body and j < num_joint and n < max_frame:
                    data[:, n, j, m] = [v["x"], v["y"], v["z"]]
    return data


class SkeletonDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.array(np.load(data_path, mmap_mode="r"), dtype=np.float32)
        with open(label_path, "rb") as f:
            self.sample_names, self.labels = pickle.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit = [int(l in rank[i, -k:]) for i, l in enumerate(label)]
    return sum(hit) / len(hit)


def plot_confusion_matrix(labels, preds, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_classes = sorted(set(labels))
    pred_classes = sorted(set(preds))

    cm = np.zeros((len(true_classes), len(pred_classes)), dtype=int)
    true_idx = {c: i for i, c in enumerate(true_classes)}
    pred_idx = {c: i for i, c in enumerate(pred_classes)}
    for t, p in zip(labels, preds):
        cm[true_idx[t], pred_idx[p]] += 1

    row_names = [f"A{c+1:03d} ({ACTION_NAMES[c]})" if c < len(ACTION_NAMES) else f"A{c+1:03d}" for c in true_classes]
    col_names = [f"A{c+1:03d} ({ACTION_NAMES[c]})" if c < len(ACTION_NAMES) and c < NUM_NEU_CLASSES else f"A{c+1:03d}" for c in pred_classes]

    fig_w = max(8, len(pred_classes) * 0.5)
    fig_h = max(6, len(true_classes) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # Annotate each cell
    for i in range(len(true_classes)):
        for j in range(len(pred_classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks(range(len(pred_classes)))
    ax.set_yticks(range(len(true_classes)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Pre-trained ST-GCN on NEU 10-class test set\n"
                 f"({len(true_classes)} true x {len(pred_classes)} predicted classes)")
    fig.colorbar(im, ax=ax, label="count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def cmd_eval():
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = SkeletonDataset(DATA_PATH, LABEL_PATH)
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f"  {len(dataset)} samples, {len(set(dataset.labels))} classes "
          f"({sorted(set(dataset.labels))})")

    print(f"Evaluating on {device}...")
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

    # Overall accuracy (60-class output vs 0-9 labels)
    top1 = topk_accuracy(scores, labels, k=1)
    top5 = topk_accuracy(scores, labels, k=5)

    # Also compute accuracy within NEU classes only (ignore predictions >= 10)
    neu_mask = preds < NUM_NEU_CLASSES
    within_neu_top1 = (preds[neu_mask] == labels[neu_mask]).float().mean().item()
    neu_ratio = neu_mask.float().mean().item()

    lines = []
    lines.append(f"{'=' * 55}")
    lines.append(f"  Pre-trained ST-GCN (NTU-60 X-Sub) on NEU 10-class test")
    lines.append(f"  Test samples: {len(labels)}")
    lines.append(f"  Top-1 Accuracy:  {top1 * 100:.2f}%")
    lines.append(f"  Top-5 Accuracy:  {top5 * 100:.2f}%")
    lines.append(f"  Predictions within NEU classes (0-9): {neu_ratio * 100:.1f}%")
    lines.append(f"  Top-1 among within-class predictions: {within_neu_top1 * 100:.2f}%")
    lines.append(f"{'=' * 55}")
    lines.append("")
    lines.append("Per-class accuracy:")
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        cls_top1 = topk_accuracy(scores[mask], labels[mask], k=1)
        cls_top5 = topk_accuracy(scores[mask], labels[mask], k=5)
        action_name = ACTION_NAMES[c] if c < len(ACTION_NAMES) else "?"
        lines.append(
            f"  Class {c:2d} (A{c + 1:03d} {action_name:<14}): "
            f"Top1={cls_top1 * 100:.1f}%  Top5={cls_top5 * 100:.1f}%  "
            f"({mask.sum().item()} samples)"
        )

    # Where does the model misclassify to?
    lines.append("")
    lines.append("Misclassification analysis (true -> predicted):")
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        wrong_mask = mask & (preds != c)
        if wrong_mask.sum() > 0:
            wrong_preds = preds[wrong_mask]
            unique, counts = torch.unique(wrong_preds, return_counts=True)
            top_wrong = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:3]
            parts = []
            for p, n in top_wrong:
                parts.append(f"A{p + 1:03d}({n})")
            action_name = ACTION_NAMES[c] if c < len(ACTION_NAMES) else "?"
            lines.append(f"  A{c + 1:03d} ({action_name}): -> {', '.join(parts)}")

    result_text = "\n".join(lines)
    print(result_text)

    os.makedirs("outputs", exist_ok=True)
    result_path = "outputs/pretrained_stgcn_eval.txt"
    with open(result_path, "w") as f:
        f.write(result_text)
    print(f"\nResults saved to {result_path}")

    # Confusion matrix (limited to NEU classes + top confused NTU classes)
    cm_path = "outputs/pretrained_stgcn_confusion.png"
    # Map predictions outside NEU range to avoid huge matrix
    clipped_preds = preds.clone()
    # Keep as-is, plot full confusion
    plot_confusion_matrix(labels.tolist(), clipped_preds.tolist(), cm_path)


def cmd_predict(skeleton_path):
    assert os.path.exists(skeleton_path), f"File not found: {skeleton_path}"

    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data = read_xyz(skeleton_path, max_body=2, num_joint=25, max_frame=300)
    tensor = torch.from_numpy(data).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor.to(device))
        scores = F.softmax(output, dim=1).squeeze(0)

    top5_scores, top5_indices = scores.topk(min(5, len(scores)))

    import re
    m = re.search(r"A(\d{3})", os.path.basename(skeleton_path))
    true_label = int(m.group(1)) - 1 if m else None
    true_name = ACTION_NAMES[true_label] if true_label is not None and true_label < len(ACTION_NAMES) else "Unknown"

    print(f"\nFile: {skeleton_path}")
    print(f"True label: [{true_label}] {true_name}")
    print(f"{'Rank':<6} {'Class':<20} {'Score':<10}")
    print("-" * 40)
    for rank, (idx, score) in enumerate(zip(top5_indices, top5_scores), 1):
        name = ACTION_NAMES[idx.item()] if idx.item() < len(ACTION_NAMES) else f"NTU-A{idx.item() + 1:03d}"
        marker = " <--" if idx.item() == true_label else ""
        print(f"  {rank:<4} [{idx.item():2d}] {name:<17} {score.item():.4f}{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-trained ST-GCN evaluation & prediction for NEU data")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("eval", help="Evaluate on NEU 10-class test set")
    pred_parser = sub.add_parser("predict", help="Predict a single .skeleton file")
    pred_parser.add_argument("path", help="Path to .skeleton file")

    args = parser.parse_args()

    if args.command == "eval":
        cmd_eval()
    elif args.command == "predict":
        cmd_predict(args.path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
