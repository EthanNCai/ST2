import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

NAME = './SPX-10.csv'

sliding_var = 30
df = pd.read_csv(NAME)
vol = df['vol']
close = df['close']

print('raw',len(close.to_list()))
def scaling(raw_data):
    scaler = preprocessing.StandardScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


def find_anomalies(data_in, steps):
    anomaly_datas = []
    anomaly_steps = []
    # Set upper and lower limit to 3 standard deviation
    data_std = np.std(data_in)
    data_mean = np.mean(data_in)
    anomaly_cut_off = data_std * 3

    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off

    for data, step in zip(data_in, steps):
        # if data > upper_limit or data < lower_limit:
        if True:
            anomaly_steps.append(step)
            anomaly_datas.append(data)
        else:
            ...
    return anomaly_datas, anomaly_steps


anomaly_datas = []
anomaly_steps = []
vol = np.array(vol.to_list())
close = np.array(close.to_list())

vol = scaling(vol)
close = scaling(close)
vol_steps = np.arange(0, len(vol), 1)

# slide through the time series
for i in range(0, len(vol_steps)):
    # np.var(vol[i-sliding_var:i])
    if i+sliding_var <= len(vol_steps):
        anomaly_datas_, anomaly_steps_ = find_anomalies(vol[i:i+sliding_var], vol_steps[i:i+sliding_var])
        anomaly_datas.extend(anomaly_datas_)
        anomaly_steps.extend(anomaly_steps_)
        print("kk",i, i+sliding_var)
anomaly_prices = np.array([close[i] for i in anomaly_steps])

print('len(anomaly_prices)', len(set(anomaly_prices)))
