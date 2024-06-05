import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy

NAME = './HSI-10.csv'

sliding_var = 10
df = pd.read_csv(NAME)
vol = df['vol']
close = df['close']

print('raw count ->', len(close.to_list()))


def scaling(raw_data):
    raw_data = np.array(raw_data)
    scaler = preprocessing.StandardScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


def calculate_outlier_boundary(data_in):
    data_in = np.array(data_in)
    data_std = np.std(data_in)
    data_mean = np.mean(data_in)
    anomaly_cut_off = data_std * 3

    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off

    return lower_limit, upper_limit

vol = scaling(vol.to_list())
vol = list(vol)
close = scaling(close.to_list())
close = list(close)

outliers_list = []
outliers_flag_list = []
outliers_index_list = []

outliers_flag_list.extend([0] * sliding_var)


for i in range(sliding_var, len(vol)):
    data = vol[i]
    lower_limit, upper_limit = calculate_outlier_boundary(vol[i - sliding_var:i])
    if upper_limit < data or lower_limit > data:
        outliers_flag_list.append(1)
        outliers_index_list.append(i)
        outliers_list.append(data)
    else:
        outliers_flag_list.append(0)

assert len(vol) == len(outliers_flag_list)
assert len(outliers_index_list) == len(outliers_list)
print('len(outliers_list)', len(outliers_list))

# ---------------------- Visualize ---------------------- #

anomaly_prices = np.array([close[i] for i in outliers_index_list])
vol_steps = np.arange(0, len(vol), 1)
anomaly_steps = outliers_index_list
fig, axes = plt.subplots(figsize=(6, 4))
# ax.plot(vol_steps, close, c='r')
axes.scatter(anomaly_steps, anomaly_prices, c='green', zorder=99, marker='x', s=15)
axes.plot(vol_steps, close, c='r', linewidth='0.8',alpha = 0.5)

axins = inset_axes(axes, width="50%", height="38%", loc='lower left',
                   borderpad=1,
                   bbox_to_anchor=(0, 0, 1, 1),
                   bbox_transform=axes.transAxes)
# axes.grid()
axins.scatter(anomaly_steps, anomaly_prices, c='green', marker='x', zorder=99, s=35)
axins.plot(vol_steps, close, c='r', linewidth='1.9')
mark_inset(axes, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1, zorder=199)

axins.set_xlim(2100, 2500)
axins.set_ylim(-0.4, 1.5)

axins.set_xticks([])
axins.set_yticks([])

plt.show()