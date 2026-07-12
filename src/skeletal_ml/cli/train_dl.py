"""
Deep learning training pipeline: AGCN / AAGCN for skeleton-based action recognition.

Usage:
    uv run skeletal-train-dl --config config/neu/train_joint.yaml
    uv run skeletal-train-dl --config config/neu/test_joint.yaml --phase test --weights <path>.pt
"""


def main():
    from skeletal_dl.trainer import main as trainer_main
    trainer_main()


if __name__ == '__main__':
    main()
