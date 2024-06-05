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

def scaling(raw_data):
    scaler = preprocessing.StandardScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)

def find_anomalies(random_data,steps):
    anomaly_datas = []
    anomaly_steps = []
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # print(lower_limit)
    # Generate outliers
    for data, step in zip(random_data, steps):
        if data > upper_limit or data < lower_limit:
            anomaly_steps.append(step)
            anomaly_datas.append(data)
        else:
            ...
    return anomaly_datas,anomaly_steps

anomaly_datas = []
anomaly_steps = []
vol = np.array(vol.to_list())
close = np.array(close.to_list())
# print(vol)

vol = scaling(vol)
close = scaling(close)
vol_steps = np.arange(0, len(vol), 1)
# ax[0].plot(vol)
# ax[1].plot(close)

for i in range(sliding_var, len(vol_steps)):
   # np.var(vol[i-sliding_var:i])
   anomaly_datas_, anomaly_steps_ = find_anomalies(vol[i-sliding_var:i], vol_steps[i-sliding_var:i])
   anomaly_datas.extend(anomaly_datas_)
   anomaly_steps.extend(anomaly_steps_)

anomaly_prices = np.array([close[i] for i in anomaly_steps])



fig, axes = plt.subplots(figsize=(6,4))
# ax.plot(vol_steps, close, c='r')
axes.scatter(anomaly_steps,anomaly_prices,c='green',zorder=99,marker = 'x', s=15)
axes.plot(vol_steps, close,c='r',linewidth='1.5')

axins = inset_axes(axes, width="50%", height="38%", loc='upper left',
                   borderpad=0,
                   bbox_to_anchor=(0, 0, 1, 1),
                   bbox_transform=axes.transAxes)
# axes.grid()
axins.scatter(anomaly_steps,anomaly_prices,c='green',marker = 'x',zorder=99, s=15)
axins.plot(vol_steps, close,c='r',linewidth='1.5')
mark_inset(axes, axins, loc1=1, loc2=3, fc="none", ec='black', lw=1,zorder=199)

axins.set_xlim(2050, 2400)
axins.set_ylim(0.5, 1.7)

axins.set_xticks([])
axins.set_yticks([])

plt.show()
