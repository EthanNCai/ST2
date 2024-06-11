import numpy as np
from sklearn import preprocessing


def scaling(raw_data):
    raw_data = np.array(raw_data)
    scaler = preprocessing.StandardScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


def find_outliers(samples_in: list, sliding_window_len: int):
    def calculate_outlier_boundary(data_in):
        data_in = np.array(data_in)
        data_std = np.std(data_in)
        data_mean = np.mean(data_in)
        anomaly_cut_off = data_std * 3

        lower_limit_ = data_mean - anomaly_cut_off
        upper_limit_ = data_mean + anomaly_cut_off

        return lower_limit_, upper_limit_

    outliers_list = []  # [1234,5343,462,3452,2432,435] len -> outliers numbers
    outliers_flag_list = []  # [0,0,0,1,1,0] len -> entire sample numbers
    outliers_index_list = []  # [1,534,6443,234] len -> outliers numbers

    outliers_flag_list.extend([0] * sliding_window_len)
    # consider the formast 30 samples normal.

    for i in range(sliding_window_len, len(samples_in)):
        data = samples_in[i]
        lower_limit, upper_limit = calculate_outlier_boundary(samples_in[i - sliding_window_len:i])
        if upper_limit < data or lower_limit > data:
            outliers_flag_list.append(1)
            outliers_index_list.append(i)
            outliers_list.append(data)
        else:
            outliers_flag_list.append(0)

    return outliers_list, outliers_flag_list, outliers_index_list


