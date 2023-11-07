from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, steps_per_day=12*24
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes, _ = data.shape
    data_list = [data]
    if add_time_in_day:
        # numerical time_of_day
        tod = [i % steps_per_day /
               steps_per_day for i in range(data.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(tod_tiled)
    
    if add_day_in_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        data.append(dow_tiled)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
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
    # read data
    data = np.load(args.traffic_df_filename)["data"]
    data = data[..., [0]]
    print("raw time series shape: {0}".format(data.shape))
    
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False
    )

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
            os.path.join(args.output_dir, "%s.npz" % cat),
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
    parser.add_argument('--dataset', type=str, choices=['PEMS04', 'PEMS08'], default='PEMS08', help='which dataset to run')
    parser.add_argument("--output_dir", type=str, default="PEMS08/", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="{}/{}.npz", help="Raw traffic readings.")
    args = parser.parse_args()
    args.output_dir = f'{args.dataset}/'
    args.traffic_df_filename = args.traffic_df_filename.format(args.dataset, args.dataset)
    main(args)
