from find_outliers import find_outliers
import pandas as pd
from sklearn import preprocessing
import numpy as np

#
# NAME = 'HSI-10'
NAME = 'IXIC-10'
sliding_var = 10
df = pd.read_csv(NAME + '.csv')
vol = df['vol']
# close = df['close']

# print('raw count ->', len(close.to_list()))


def scaling(raw_data):
    raw_data = np.array(raw_data)
    scaler = preprocessing.StandardScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


vol = scaling(vol.to_list())
vol = list(vol)
# close = scaling(close.to_list())
# close = list(close)

outliers_list, outliers_flag_list, outliers_index_list = find_outliers(vol, sliding_var)

assert len(vol) == len(outliers_flag_list)
assert len(outliers_index_list) == len(outliers_list)
print('len(outliers_list) -> ', len(outliers_list))

outliers_flag_series = pd.Series(outliers_flag_list)

df['is_vol_outliers'] = outliers_flag_series

df.to_csv(NAME + '-VOF.csv')
