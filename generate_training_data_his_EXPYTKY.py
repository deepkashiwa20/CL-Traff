import os
import numpy as np
import pandas as pd
import scipy.sparse as ss
import argparse
from configparser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('--month', type=str, default='202112', help='In gen_dataplus, it must be set to 202112.')
opt = parser.parse_args()
config = ConfigParser()
config.read('./EXPYTKY/params.txt', encoding='UTF-8')
train_month = eval(config[opt.month]['train_month'])
test_month = eval(config[opt.month]['test_month'])
traffic_path = config[opt.month]['traffic_path']
road_path = config['common']['road_path']
N_link = int(config['common']['N_link'])
df_train_avg = config['common']['history_average']

def generate_graph_seq2seq_io_data_with_his(data, x_offsets, y_offsets):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

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


def generate_train_val_test(train_month, months, months_path, road_path):
    if not os.path.exists(df_train_avg):
        df_train = pd.concat([pd.read_csv(months_path[month]) for month in train_month])
        df_train.loc[df_train['speed']<0, 'speed'] = 0
        df_train.loc[df_train['speed']>200, 'speed'] = 100
        
        df_train['timeinday'] = (pd.to_datetime(df_train['timestamp'].values) - pd.to_datetime(df_train['timestamp'].values.astype("datetime64[D]"))) / np.timedelta64(1, "D")
        
        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
        df_train['weekdaytime'] = df_train['timestamp'].dt.weekday * 144 + (df_train['timestamp'].dt.hour * 60 + df_train['timestamp'].dt.minute)//10
        
        data = df_train[['linkid', 'weekdaytime', 'speed']]
        
        def get_mean_without_null_values(data, null_val=0.):
            return data[data != null_val].mean()
        df_train_avg = data.groupby(['linkid', 'weekdaytime']).aggregate({'speed': get_mean_without_null_values}).reset_index()
        # df_train_avg = data.groupby(['linkid', 'weekdaytime']).mean().reset_index()
    else:
        df_train_avg = pd.read_csv(df_train_avg)
    
    for month in months:
        df_test = pd.read_csv(months_path[month])
        df_test.loc[df_test['speed']<0, 'speed'] = 0
        df_test.loc[df_test['speed']>200, 'speed'] = 100
        
        df_test['timeinday'] = (pd.to_datetime(df_test['timestamp'].values) - pd.to_datetime(df_test['timestamp'].values.astype("datetime64[D]"))) / np.timedelta64(1, "D")
        
        df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
        df_test['weekdaytime'] = df_test['timestamp'].dt.weekday * 144 + (df_test['timestamp'].dt.hour * 60 + df_test['timestamp'].dt.minute)//10

        df = pd.merge(df_test, df_train_avg, on=['linkid', 'weekdaytime'], suffixes=(None, '_y'))
        
        df_capital_link = pd.read_csv(road_path)
        capital_linkid_list = df_capital_link['link_id'].unique()
        timeslices = df_test.timestamp.unique() # must be datetime
        mux = pd.MultiIndex.from_product([timeslices, capital_linkid_list],names = ['timestamp', 'linkid'])
        df = df.set_index(['timestamp', 'linkid']).reindex(mux).reset_index()
        data_his = df[['timestamp', 'linkid', 'speed', 'timeinday', 'speed_y']]
        
        data_his = data_his[['speed', 'timeinday', 'speed_y']]
        data_his.to_csv(f'./EXPYTKY/expy-tky_plus_{month}.csv.gz', index=False)
        
        data_his = data_his.values.reshape(-1, N_link, 3)
        print('generate capital traffic plus over', month, data_his.shape)
    
    # TODO
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data_with_his(
        data_his,
        x_offsets=x_offsets,
        y_offsets=y_offsets
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
            os.path.join(args.output_dir, "%shis.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    if not os.path.exists(config[opt.month]['trafficplus_path']):
        months = train_month + test_month
        months_path = {month:config[month]['traffic_path'] for month in months}
        print(f'train_month:{train_month}, test_month:{test_month}, months:{months}')
        generate_train_val_test(train_month, months, months_path, road_path)
