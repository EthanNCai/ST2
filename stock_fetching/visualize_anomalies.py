import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
from find_outliers import find_outliers

NAME = './HSI-10.csv'
# NAME = './AIR.csv'
sliding_var = 10
df = pd.read_csv(NAME)
vol = df['vol']
# close = df['close']

print('raw count ->', len(close.to_list()))


def scaling(raw_data):
    raw_data = np.array(raw_data)
    scaler = preprocessing.StandardScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


vol = scaling(vol.to_list())
vol = list(vol)
close = scaling(close.to_list())
close = list(close)

outliers_list, outliers_flag_list, outliers_index_list = find_outliers(vol, sliding_var)

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