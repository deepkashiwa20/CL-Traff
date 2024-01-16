import os
import numpy as np
import pandas as pd
import scipy.sparse as ss
import argparse
from configparser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('--month', type=str, default='202112', help='In gen_dataplus, it must be set to 202112.')
parser.add_argument('--dataset', type=str, choices=['EXPYTKY', 'EXPYTKY*'], default='EXPYTKY', help='which dataset to run')
parser.add_argument('--val_ratio', type=float, default=0.25, help='the ratio of validation data among the trainval ratio')
parser.add_argument("--output_dir", type=str, default="EXPYTKY/", help="Output directory.")
opt = parser.parse_args()
config = ConfigParser()
config.read('./EXPYTKY/params.txt', encoding='UTF-8')
train_month = eval(config[opt.month]['train_month'])
test_month = eval(config[opt.month]['test_month'])
traffic_path = config[opt.month]['traffic_path']
road_path = config['common']['road_path']
subroad_path = config[opt.dataset]['subroad_path']
N_link = int(config['common']['N_link'])
train_avg_path = config['common']['history_average']

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
    if not os.path.exists(train_avg_path):
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
        df_train_avg.to_csv(train_avg_path, index=False)
    else:
        df_train_avg = pd.read_csv(train_avg_path)
    
    train_data = []
    test_data = []
    for month in months:
        # if not os.path.exists(f'./EXPYTKY/expy-tky_plus_{month}.csv.gz', index=False):
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
        # data_his.to_csv(f'./EXPYTKY/expy-tky_plus_{month}.csv.gz', index=False)
        # else:
        #     data_his = pd.read_csv(f'./EXPYTKY/expy-tky_plus_{month}.csv.gz')     #* 存储于csv.gz文件下的timeinday会存在精度误差
            
        data_his = data_his.values.reshape(-1, N_link, 3)
        print('generate capital traffic plus over', month, data_his.shape)
        sub_idx = np.loadtxt(subroad_path).astype(int)
        data_his = data_his[:, sub_idx, :]
        print(f're-generate {opt.dataset} traffic plus over', month, data_his.shape)
        if month in train_month:
            train_data.append(data_his)
        else:
            test_data.append(data_his)
            
    trainval_data = np.vstack(train_data)
    test_data = np.vstack(test_data)
    
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-5, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 7, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    trainXS, trainYS = generate_graph_seq2seq_io_data_with_his(
        trainval_data,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )

    print("Train XS shape: ", trainXS.shape, ", Train YS shape: ", trainYS.shape)
    
    testXS, testYS = generate_graph_seq2seq_io_data_with_his(
        test_data,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )

    print("Test XS shape: ", testXS.shape, ", Test YS shape: ", testYS.shape)
    
    trainval_size = len(trainXS)
    train_size = int(trainval_size * (1 - opt.val_ratio))
    
    # train
    x_train, y_train = trainXS[:train_size], trainYS[:train_size]
    # val
    x_val, y_val = trainXS[train_size:], trainYS[train_size:]
    # test
    x_test, y_test = testXS, testYS

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(opt.output_dir, "%shis.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    if not os.path.exists(config[opt.month]['traffichis_path']):
        months = train_month + test_month
        months_path = {month:config[month]['traffic_path'] for month in months}
        print(f'train_month:{train_month}, test_month:{test_month}, months:{months}')
        generate_train_val_test(train_month, months, months_path, road_path)
