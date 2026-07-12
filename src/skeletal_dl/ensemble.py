import argparse
import pickle

import numpy as np
from tqdm import tqdm


def ensemble(dataset='neu', alpha=1.0):
    label_path = f'./data/{dataset}/val_label.pkl'
    with open(label_path, 'rb') as f:
        label = np.array(pickle.load(f))

    joint_score_path = f'./work_dir/{dataset}/agcn_test_joint/epoch1_test_score.pkl'
    bone_score_path = f'./work_dir/{dataset}/agcn_test_bone/epoch1_test_score.pkl'

    r1 = open(joint_score_path, 'rb')
    r1 = list(pickle.load(r1).items())
    r2 = open(bone_score_path, 'rb')
    r2 = list(pickle.load(r2).items())

    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        r = r11 + r22 * alpha
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(f'Top1: {acc:.4f}, Top5: {acc5:.4f}')
    return acc, acc5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default='neu',
                        help='dataset name')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='weighted summation')
    args = parser.parse_args()
    ensemble(args.datasets, args.alpha)


if __name__ == '__main__':
    main()
