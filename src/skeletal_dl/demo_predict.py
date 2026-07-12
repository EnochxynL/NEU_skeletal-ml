"""
3D skeleton action recognition demo using a DL model (ST-GCN / AGCN / ST-GIN).

Shows static multi-frame view + optional animated GIF with predicted label.

Usage:
    # Using pre-trained ST-GCN (default)
    uv run python src/skeletal_dl/demo_predict.py data/test/S001C001P001R001A005.skeleton

    # Using a trained checkpoint with different model
    uv run python src/skeletal_dl/demo_predict.py data/test/S001C001P001R001A005.skeleton \
        --checkpoint runs/neu_stgcn_joint-39-2730.pt \
        --model skeletal_dl.model.stgcn.Model \
        --num-class 10

    # Save animated GIF
    uv run python src/skeletal_dl/demo_predict.py data/test/S001C001P001R001A005.skeleton --gif
"""

import argparse
import os
import re
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -- skeleton reader -----------------------------------------------------------

BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7),
    (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11),
    (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

ACTION_NAMES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop",
    "pick up", "throw", "sit down", "stand up", "clapping",
]


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


# -- model loading -------------------------------------------------------------

def _import_class(name):
    import importlib
    module_path, class_name = name.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def build_stgcn_pretrained():
    """Load the NTU-60 X-Sub pre-trained ST-GCN."""
    from skeletal_dl.model.stgcn import Model
    from skeletal_dl.graph.ntu_rgb_d import Graph

    model = Model(
        num_class=60, num_point=25, num_person=2,
        graph="skeletal_dl.graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
    )
    ckpt_path = "D:/ACTIVE/NEUAI/mmskeleton/checkpoints/st_gcn.ntu-xsub-300b57d4.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, 60


def build_model_from_checkpoint(model_path, checkpoint_path, num_class, **model_args):
    """Load an arbitrary model from a training checkpoint."""
    ModelCls = _import_class(model_path)
    model = ModelCls(num_class=num_class, num_point=25, num_person=2, **model_args)
    weights = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model, num_class


# -- prediction ----------------------------------------------------------------

def predict(model, num_class, filepath, device):
    data = read_xyz(filepath, max_body=2, num_joint=25, max_frame=300)
    tensor = torch.from_numpy(data).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor.to(device))
        probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
    return probs


# -- visualization -------------------------------------------------------------

def show_skeleton_static(model, num_class, filepath, device):
    """Save a static multi-frame PNG showing key poses + prediction."""
    coords = read_xyz(filepath, max_body=2, num_joint=25, max_frame=300)
    coords = coords[:, :, :, 0].transpose(1, 2, 0)  # (T, 25, 3)
    T = coords.shape[0]

    probs = predict(model, num_class, filepath, device)
    pred_label = int(np.argmax(probs))
    pred_name = ACTION_NAMES[pred_label] if pred_label < len(ACTION_NAMES) else f"Class {pred_label}"

    m = re.search(r"A(\d{3})", os.path.basename(filepath))
    true_label = int(m.group(1)) - 1 if m else -1
    true_name = ACTION_NAMES[true_label] if 0 <= true_label < len(ACTION_NAMES) else "Unknown"
    correct = pred_label == true_label

    # Center + rotate for display
    coords = coords - coords[:, 0:1, :]
    coords = coords[..., [0, 2, 1]]
    coords[..., 1] *= -1

    num_frames = min(4, T)
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 5),
                             subplot_kw={"projection": "3d"})

    status = "CORRECT" if correct else "WRONG"
    color = "green" if correct else "red"
    fig.suptitle(
        f"Pred: [{pred_label}] {pred_name}  |  "
        f"True: [{true_label}] {true_name}  |  {status}",
        fontsize=14, color=color, fontweight="bold")

    xall, yall, zall = coords[..., 0].ravel(), coords[..., 1].ravel(), coords[..., 2].ravel()
    pad = 0.2

    for ax, idx in zip(axes if num_frames > 1 else [axes], indices):
        pts = coords[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=20)
        for j1, j2 in BONES:
            ax.plot([pts[j1, 0], pts[j2, 0]],
                    [pts[j1, 1], pts[j2, 1]],
                    [pts[j1, 2], pts[j2, 2]], c="steelblue", lw=1.5)
        ax.set_xlim(xall.min() - pad, xall.max() + pad)
        ax.set_ylim(yall.min() - pad, yall.max() + pad)
        ax.set_zlim(zall.min() - pad, zall.max() + pad)
        ax.view_init(elev=10, azim=-90)
        ax.set_title(f"Frame {idx + 1}/{T}", fontsize=10)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    base = os.path.basename(filepath).replace(".skeleton", "_dl_pred.png")
    out = os.path.join("outputs", base)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Static frame saved: {out}")

    return pred_label, pred_name, true_label, true_name, correct, probs


def show_skeleton_3d(model, num_class, filepath, device, save_gif=False):
    """Save / show an animated 3D skeleton sequence."""
    data = read_xyz(filepath, max_body=2, num_joint=25, max_frame=300)
    coords = data[:, :, :, 0].transpose(1, 2, 0)  # (T, 25, 3)
    T = coords.shape[0]

    probs = predict(model, num_class, filepath, device)
    pred_label = int(np.argmax(probs))
    pred_name = ACTION_NAMES[pred_label] if pred_label < len(ACTION_NAMES) else f"Class {pred_label}"

    m = re.search(r"A(\d{3})", os.path.basename(filepath))
    true_label = int(m.group(1)) - 1 if m else -1
    true_name = ACTION_NAMES[true_label] if 0 <= true_label < len(ACTION_NAMES) else "Unknown"
    correct = pred_label == true_label

    coords = coords - coords[:, 0:1, :]
    coords = coords[..., [0, 2, 1]]
    coords[..., 1] *= -1

    xmin, xmax = coords[:, :, 0].min(), coords[:, :, 0].max()
    ymin, ymax = coords[:, :, 1].min(), coords[:, :, 1].max()
    zmin, zmax = coords[:, :, 2].min(), coords[:, :, 2].max()
    pad = 0.3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    lines = []
    scatter = None

    def init():
        nonlocal scatter
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_zlim(zmin - pad, zmax + pad)
        ax.view_init(elev=10, azim=-90)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        status = "CORRECT" if correct else "WRONG"
        color = "green" if correct else "red"
        ax.set_title(
            f"Pred: [{pred_label}] {pred_name} | "
            f"True: [{true_label}] {true_name} | {status}",
            fontsize=13, color=color, fontweight="bold")
        scatter = ax.scatter([], [], [], c="red", s=30)
        for _ in BONES:
            line, = ax.plot([], [], [], c="steelblue", lw=1.5)
            lines.append(line)
        return [scatter] + lines

    def update(frame):
        pts = coords[frame]
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        for i, (j1, j2) in enumerate(BONES):
            lines[i].set_data_3d(
                [pts[j1, 0], pts[j2, 0]],
                [pts[j1, 1], pts[j2, 1]],
                [pts[j1, 2], pts[j2, 2]])
        status = "CORRECT" if correct else "WRONG"
        color = "green" if correct else "red"
        ax.set_title(
            f"Pred: [{pred_label}] {pred_name} | "
            f"True: [{true_label}] {true_name} | {status} | "
            f"Frame {frame + 1}/{T}",
            fontsize=13, color=color, fontweight="bold")
        return [scatter] + lines

    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                        interval=50, blit=False, repeat=True)

    if save_gif:
        os.makedirs("outputs", exist_ok=True)
        base = os.path.basename(filepath).replace(".skeleton", "_dl_pred.gif")
        out = os.path.join("outputs", base)
        ani.save(out, writer="pillow", fps=20, dpi=80)
        print(f"GIF saved: {out}")
    else:
        plt.show()


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DL model 3D skeleton action recognition demo")
    parser.add_argument("filepath", help="Path to .skeleton file")
    parser.add_argument("--gif", action="store_true", help="Save animated GIF")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--model", default=None,
                        help="Model class path (e.g. skeletal_dl.model.stgcn.Model)")
    parser.add_argument("--num-class", type=int, default=10,
                        help="Number of classes (for trained model)")
    parser.add_argument("--graph", default="skeletal_dl.graph.ntu_rgb_d.Graph",
                        help="Graph class path")
    parser.add_argument("--labeling-mode", default="spatial",
                        help="Graph labeling mode")

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"File not found: {args.filepath}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.checkpoint is not None and args.model is not None:
        print(f"Loading model: {args.model}")
        print(f"Checkpoint: {args.checkpoint}")
        model, num_class = build_model_from_checkpoint(
            args.model, args.checkpoint, args.num_class,
            graph=args.graph,
            graph_args={"labeling_mode": args.labeling_mode},
        )
    else:
        print("Loading pre-trained ST-GCN (NTU-60 X-Sub)...")
        model, num_class = build_stgcn_pretrained()

    model = model.to(device)
    print(f"Model loaded. {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params\n")

    # Static multi-frame
    pred_label, pred_name, true_label, true_name, correct, probs = \
        show_skeleton_static(model, num_class, args.filepath, device)

    # Print results
    print(f"\n  Prediction: [{pred_label}] {pred_name}")
    print(f"  True label: [{true_label}] {true_name}")
    print(f"  Correct: {'Yes' if correct else 'No'}")
    print(f"  Top-5 probabilities:")
    top5 = np.argsort(probs)[::-1][:5]
    for rank, idx in enumerate(top5):
        name = ACTION_NAMES[idx] if idx < len(ACTION_NAMES) else f"NTU-A{idx + 1:03d}"
        marker = " <--" if idx == pred_label else ""
        print(f"    {rank + 1}. [{idx:2d}] {name:<17} {probs[idx]:.4f}{marker}")
    print()

    # 3D animation
    try:
        show_skeleton_3d(model, num_class, args.filepath, device, save_gif=args.gif)
    except Exception as e:
        print(f"(Could not show 3D animation: {e})")


if __name__ == "__main__":
    main()
