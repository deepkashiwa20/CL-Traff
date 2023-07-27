"""
compute the history average (HA) and the corresponding statistics analysis
"""
import pandas as pd
import numpy as np
import os, sys
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
import torch
sys.path.append("..")
from model.utils import masked_mae_loss, masked_mape_loss, masked_rmse_loss

sns.set_theme(style="darkgrid")
plt.figure()
dir_name = os.path.dirname(os.path.abspath(__file__))
TIMESLOT = 288 * 7
EXCEPTION_THRESHOLDS = [0.1, 0.5, 1, 5, 10, 15, 20, 25]
EXCEPTION_THRESHOLDS_WITH_MINMAX_TRANSFORMATION = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EXCEPTION_THRESHOLDS_WITH_STANDARD_TRANSFORMATION = [0.01, 0.1, 1, 4, 9, 16]

def historical_average_predict(df, period=12 * 24 * 7, test_ratio=0.2, null_val=0.):
    """
    Calculates the historical average of sensor reading.
    :param df:
    :param period: default 1 week.
    :param test_ratio:
    :param null_val: default 0.
    :return:
    """
    n_sample, n_sensor = df.shape  # (34272, 207)
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test  # 27418
    y_test = df[-n_test:]  # (6854, 207)
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):  # range(27418, 29434)
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]  # (13, 207)
        y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test

def eval_historical_average(traffic_reading_df, period=7*24*12):
    y_predict, y_test = historical_average_predict(traffic_reading_df, period=period, test_ratio=0.2)
    y_predict, y_test = torch.from_numpy(y_predict.values), torch.from_numpy(y_test.values)
    rmse = masked_rmse_loss(y_predict, y_test)
    mape = masked_mape_loss(y_predict, y_test)
    mae = masked_mae_loss(y_predict, y_test)
    print('Historical Average (DCRNN Baseline)')
    print('\t'.join(['Model', 'Horizon', 'MAE', 'RMSE', 'MAPE']))
    for horizon in [1, 3, 6, 12]:
        line = 'HA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, mae, rmse, mape * 100)
        print(line)
        
def evaluate_HA(ys_pred, ys_true):
    ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)  # (T, B, N, C)
    mae = masked_mae_loss(ys_pred, ys_true).item()
    mape = masked_mape_loss(ys_pred, ys_true).item()
    rmse = masked_rmse_loss(ys_pred, ys_true).item()
    mae_3 = masked_mae_loss(ys_pred[2:3], ys_true[2:3]).item()
    mape_3 = masked_mape_loss(ys_pred[2:3], ys_true[2:3]).item()
    rmse_3 = masked_rmse_loss(ys_pred[2:3], ys_true[2:3]).item()
    mae_6 = masked_mae_loss(ys_pred[5:6], ys_true[5:6]).item()
    mape_6 = masked_mape_loss(ys_pred[5:6], ys_true[5:6]).item()
    rmse_6 = masked_rmse_loss(ys_pred[5:6], ys_true[5:6]).item()
    mae_12 = masked_mae_loss(ys_pred[11:12], ys_true[11:12]).item()
    mape_12 = masked_mape_loss(ys_pred[11:12], ys_true[11:12]).item()
    rmse_12 = masked_rmse_loss(ys_pred[11:12], ys_true[11:12]).item()
    print('Horizon overall: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}'.format(mae, rmse, mape * 100))
    print('Horizon 15mins: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}'.format(mae_3, rmse_3, mape_3 * 100))
    print('Horizon 30mins: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}'.format(mae_6, rmse_6, mape_6 * 100))
    print('Horizon 60mins: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}'.format(mae_12, rmse_12, mape_12 * 100))
    ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)

def extract_data(args):
    df = pd.read_csv(os.path.join(dir_name, "..", args.traffic_df_filename), compression='gzip')
    data = df[['sensorid', 'weekdaytime', 'speed', 'speed_y']]
    data['diff'] = (data['speed'] - data['speed_y']).abs()  # (34272 * 207, 5)
    return data 

def get_history_average(args):
    data = extract_data(args)
    num_nodes = args.num_nodes
    num_samples = data.shape[0] // num_nodes
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    y_offsets = np.sort(np.arange(1, 13, 1))
    preds, labels = [], []
    data_speed = data['speed'].values.reshape(num_nodes, -1, 1).transpose(1, 0, 2)  # (34272, 207, 1)
    data_speedy = data['speed_y'].values.reshape(num_nodes, -1, 1).transpose(1, 0, 2) 
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        pred_t = data_speedy[t + y_offsets, ...]  #* 用当前的平均速度还是1小时后的平均速度呢?
        label_t = data_speed[t + y_offsets, ...]
        preds.append(pred_t) 
        labels.append(label_t)
        
    preds = np.stack(preds, axis=0)  # (34249, 12, 207, 1)
    labels = np.stack(labels, axis=0)  
    
    num_test = round(labels.shape[0] * 0.2)
    test_preds, test_labels = torch.from_numpy(preds[-num_test:]), torch.from_numpy(labels[-num_test:])
    evaluate_HA(test_preds, test_labels)
    
    # num_test = round(num_samples * 0.2)
    # preds, labels = data_speedy[-num_test:], data_speed[-num_test:]
    # test_preds, test_labels = torch.from_numpy(preds), torch.from_numpy(labels)
    # rmse = masked_rmse_loss(test_preds, test_labels)
    # mape = masked_mape_loss(test_preds, test_labels)
    # mae = masked_mae_loss(test_preds, test_labels)
    # print('Historical Average')
    # print('\t'.join(['Model', 'Horizon', 'MAE', 'RMSE', 'MAPE']))
    # for horizon in [1, 3, 6, 12]:
    #     line = 'HA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, mae, rmse, mape * 100)
    #     print(line)
    
def get_statistics_analysis(args):
    data = extract_data(args)
    num_nodes = args.num_nodes
    data_exception = data['diff'].values.reshape(-1, num_nodes) # (34272, 207)
    data_time = (data['weekdaytime'].values.reshape(-1, num_nodes) * TIMESLOT).astype(np.int32) # (34272, 207)
    num_samples = data_exception.shape[0]
    num_train = round(num_samples * 0.6)
    num_test = round(num_samples * 0.2)
    train_data, test_data = data_exception[:num_train], data_exception[-num_test:]
    train_time, test_time = data_time[:num_train], data_time[-num_test:]
    train_exception_mean, test_exception_mean = train_data.mean(), test_data.mean()
    train_exception_std, test_exception_std = train_data.std(), test_data.std()
    train_exception_max, test_exception_max = train_data.max(), test_data.max()
    train_exception_min, test_exception_min = train_data.min(), test_data.min()
    print('Train Statistics: mean: {:.2f}, std: {:.2f}'.format(train_exception_mean, train_exception_std))  # 9.66, 13.61
    print('Test Statistics: mean: {:.2f}, std: {:.2f}'.format(test_exception_mean, test_exception_std))  # 12.73, 17.46
    print('Train Statistics: max: {:.2f}, min: {:.2f}'.format(train_exception_max, train_exception_min))
    print('Test Statistics: max: {:.2f}, min: {:.2f}'.format(test_exception_max, test_exception_min))
    train_counts, test_counts = [], []
    if args.norm == 0:
        thresholds = EXCEPTION_THRESHOLDS
    elif args.norm == 1:
        thresholds = EXCEPTION_THRESHOLDS_WITH_MINMAX_TRANSFORMATION
    elif args.norm == 2:
        thresholds = EXCEPTION_THRESHOLDS_WITH_STANDARD_TRANSFORMATION
    else:
        raise NotImplementedError
    for threshold in thresholds:
        if args.norm == 0:
            train_count = (train_data > threshold).sum() / train_data.size * 100
            test_count = (test_data > threshold).sum() / test_data.size * 100
        elif args.norm == 1:
            train_count = (((train_data - train_exception_min) / (train_exception_max - train_exception_min)) > threshold).sum() / train_data.size * 100
            test_count = (((test_data - test_exception_min) / (test_exception_max - test_exception_min)) > threshold).sum() / test_data.size * 100
        elif args.norm == 2:
            train_count = (np.square(((train_data - train_exception_mean) / train_exception_std)) > threshold).sum() / train_data.size * 100
            test_count = (np.square(((test_data - test_exception_mean) / test_exception_std)) > threshold).sum() / test_data.size * 100
        train_counts.append(train_count)
        test_counts.append(test_count)
    train_metrics = np.stack([np.array(thresholds), np.array(train_counts)], axis=-1)
    test_metrics = np.stack([np.array(thresholds), np.array(test_counts)], axis=-1)
    train_metrics = pd.DataFrame(train_metrics, columns=['Threshold', 'Percentage'])
    test_metrics = pd.DataFrame(test_metrics, columns=['Threshold', 'Percentage'])
    save_dir = args.output_dir
    sns.relplot(data=train_metrics, x="Threshold", y='Percentage', kind="line", errorbar=None)
    for x, y in zip(train_metrics["Threshold"], train_metrics["Percentage"]):
        plt.annotate(f'{y:.2f}', (x,y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.savefig(os.path.join(save_dir, 'train_exceptions_norm{}.pdf'.format(args.norm)))
    sns.relplot(data=test_metrics, x="Threshold", y='Percentage', kind="line", errorbar=None)
    for x, y in zip(test_metrics["Threshold"], test_metrics["Percentage"]):
        plt.annotate(f'{y:.2f}', (x,y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.savefig(os.path.join(save_dir, 'test_exceptions_norm{}.pdf'.format(args.norm)))

def main(args):
    print("Visualizing statistics analysis...")
    traffic_reading_df = pd.read_hdf(os.path.join(dir_name, "..", args.traffic_reading_filename))
    eval_historical_average(traffic_reading_df, period=7 * 24 * 12)
    get_history_average(args)
    get_statistics_analysis(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
    parser.add_argument("--output_dir", type=str, default="METRLA/", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="METRLA/metr-la.csv.gz", help="Raw traffic readings.")
    parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
    parser.add_argument('--traffic_reading_filename', default="METRLA/metr-la.h5", type=str, help='Path to the traffic Dataframe.')
    parser.add_argument('--norm', type=int, default=2, choices=[0, 1, 2], help='which normalization to apply (0 means no normalization)')
    args = parser.parse_args()
    args.output_dir = os.path.join(dir_name, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset == 'METRLA':
        args.traffic_df_filename = f'{args.dataset}/metr-la.csv.gz'
        args.traffic_reading_filename = f'{args.dataset}/metr-la.h5'
        args.num_nodes = 207
    elif args.dataset == 'PEMSBAY':
        args.traffic_df_filename = f'{args.dataset}/pems-bay.csv.gz'
        args.traffic_reading_filename = f'{args.dataset}/pems-bay.h5'
        args.num_nodes = 325
    main(args)