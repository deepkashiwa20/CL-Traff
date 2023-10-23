import pandas as pd
import numpy as np
import datetime

df = pd.HDFStore('./metr-la.h5')['/df']

timeslots = pd.date_range('2012-03-01 00:00:00', '2012-06-27 23:55:00', freq='5min')
print(len(timeslots))
days = pd.date_range('2012-03-01 00:00:00', '2012-06-27 23:55:00', freq='1D')
print(len(days), 0.8*len(days))
train_days = pd.date_range('2012-03-01 00:00:00', '2012-06-03 23:55:00', freq='1D')
print(len(train_days))
print(len(timeslots), timeslots[0], timeslots[-1])
list1 = [t.strftime('%Y-%m-%d %H:%M:%S')  for t in timeslots]
print(len(list1))
list2 = [pd.to_datetime(str(x)).strftime('%Y-%m-%d %H:%M:%S')  for x in df.index.values]
print(len(list2))
print(set(list1) - set(list2))

data = df.stack().reset_index()
data.columns = ['timestamp', 'sensorid', 'speed']

data['timeinday'] = (data['timestamp'].values - data['timestamp'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['weekdaytime'] = data['timestamp'].dt.weekday * 288 + (data['timestamp'].dt.hour * 60 + data['timestamp'].dt.minute)//5

df_train = data[(data.timestamp >= '2012-03-01 00:00:00') & (data.timestamp <= '2012-06-03 23:55:00')]
df_train = df_train[['sensorid', 'weekdaytime', 'speed']]

def get_mean_without_null_values(data, null_val=0.):
    return data[data != null_val].mean()
df_train_avg = df_train.groupby(['sensorid', 'weekdaytime']).aggregate({'speed': get_mean_without_null_values}).reset_index()

data_plus = pd.merge(data, df_train_avg, on=['sensorid', 'weekdaytime'], suffixes=(None, '_y'))

sensorid_list = df.columns
timeslices = df.index
mux = pd.MultiIndex.from_product([timeslices, sensorid_list],names=['timestamp', 'sensorid'])
data_plus = data_plus.set_index(['timestamp', 'sensorid']).reindex(mux).reset_index()
data_plus = data_plus[['timestamp', 'sensorid', 'speed', 'timeinday', 'speed_y']]

np.array_equal(data_plus['speed'].values.reshape(-1, 207), df.values)

store = pd.HDFStore("metr-la-plus.h5", 'w')
store.put('/df', data_plus)

df_reload = pd.HDFStore('../METRLA/metr-la-plus.h5')['/df']

np.array_equal(df_reload['speed'].values.reshape(-1, 207), df.values)