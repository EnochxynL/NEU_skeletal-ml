import argparse
import numpy as np
from numpy.lib.format import open_memmap

paris = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
    (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
)


def gen_bone(data_dir='data/neu/'):
    sets = ['train', 'val']
    for set_name in sets:
        print(f'Generating bone data for {set_name}...')
        data = np.load(f'{data_dir}/{set_name}_data_joint.npy')
        N, C, T, V, M = data.shape

        fp_sp = open_memmap(
            f'{data_dir}/{set_name}_data_bone.npy',
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        for v1, v2 in paris:
            v1 -= 1
            v2 -= 1
            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

        print(f'  {set_name}_data_bone.npy saved.')
    print('Done.')


def main():
    parser = argparse.ArgumentParser(description='Generate bone data from joint data.')
    parser.add_argument('--data_dir', default='data/neu/', help='path to neu data directory')
    args = parser.parse_args()
    gen_bone(args.data_dir)


if __name__ == '__main__':
    main()
