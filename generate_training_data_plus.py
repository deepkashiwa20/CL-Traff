from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
    """
    Generate samples from
    :param data:
    :param x_offsets:
    :param y_offsets:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    num_samples = data.shape[0]
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.data_file)
    data = df[['speed', 'timeinday', 'speed_y']].values
    data = data.reshape(-1, 207, 3)
    
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-11, 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(data, x_offsets=x_offsets, y_offsets=y_offsets)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]


    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%splus.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
    parser.add_argument("--output_dir", type=str, default="METRLA/", help="Output directory.")
    parser.add_argument("--data_file", type=str, default="METRLA/metr-la-plus.h5", help="Raw traffic readings.")
    args = parser.parse_args()
    args.output_dir = f'{args.dataset}/'
    if args.dataset == 'METRLA':
        args.data_file = f'{args.dataset}/metr-la-plus.h5'
    elif args.dataset == 'PEMSBAY':
        args.data_file = f'{args.dataset}/pems-bay-plus.h5'
    main(args)
