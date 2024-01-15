import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import argparse
from configparser import ConfigParser
import logging
        
parser = argparse.ArgumentParser()
parser.add_argument('--month', type=str, default='202112', help='In gen_dataplus, it must be set to 202112.')
opt = parser.parse_args()
config = ConfigParser()
config.read('params.txt', encoding='UTF-8')
train_month = eval(config[opt.month]['train_month'])
test_month = eval(config[opt.month]['test_month'])
traffic_path = config[opt.month]['traffic_path']
road_path = config['common']['road_path']

# training months are used for calculating the avg speed for all months.

def generate_data_plus(train_month, months, months_path, road_path):
    df_train = pd.concat([pd.read_csv(months_path[month]) for month in train_month])
    df_train.loc[df_train['speed']<0, 'speed'] = 0
    df_train.loc[df_train['speed']>200, 'speed'] = 100
    
    df_train['timeinday'] = (df_train['timestamp'].values - df_train['timestamp'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    df_train['weekdaytime'] = df_train['timestamp'].dt.weekday * 144 + (df_train['timestamp'].dt.hour * 60 + df_train['timestamp'].dt.minute)//10
    
    df_train = df_train[['linkid', 'weekdaytime', 'speed']]
    
    def get_mean_without_null_values(data, null_val=0.):
        return data[data != null_val].mean()
    df_train_avg = df_train.groupby(['sensorid', 'weekdaytime']).aggregate({'speed': get_mean_without_null_values}).reset_index()
        
    for month in months:
        df_test = pd.read_csv(months_path[month])
        df_test.loc[df_test['speed']<0, 'speed'] = 0
        df_test.loc[df_test['speed']>200, 'speed'] = 100
        
        df_test['timeinday'] = (df_test['timestamp'].values - df_test['timestamp'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        
        df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
        df_test['weekdaytime'] = df_test['timestamp'].dt.weekday * 144 + (df_test['timestamp'].dt.hour * 60 + df_test['timestamp'].dt.minute)//10

        df = pd.merge(df_test, df_train_avg, on=['linkid', 'weekdaytime'], suffixes=(None, '_y'))
        
        df_capital_link = pd.read_csv(road_path)
        capital_linkid_list = df_capital_link['link_id'].unique()
        timeslices = df_test.timestamp.unique() # must be datetime
        mux = pd.MultiIndex.from_product([timeslices, capital_linkid_list],names = ['timestamp', 'linkid'])
        df = df.set_index(['timestamp', 'linkid']).reindex(mux).reset_index()
        df['weekdaytime'] = df['weekdaytime']/df['weekdaytime'].max()
    
        df.to_csv(f'./EXPYTKY/expy-tky_plus_{month}.csv.gz', index=False)
        print('generate capital traffic plus over', month, df.shape)
        
def main():
    if not os.path.exists(config[opt.month]['trafficplus_path']):
        months = train_month + test_month
        months_path = {month:config[month]['traffic_path'] for month in months}
        print('train_month, test_month, months', train_month, test_month, months)
        generate_data_plus(train_month, months, months_path, road_path)

if __name__ == '__main__':
    main()