import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn import preprocessing


def scaling(raw_data):
    scaler = preprocessing.MinMaxScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


class MultiFeatureDataset(Dataset):
    def __init__(self, raw_serials, time_step, target_mean_len, features, to_tensor=True):
        self.time_step = time_step
        self.target_mean_len = target_mean_len
        self.stepped_serial_data = [self.reshape_data(raw_serial) for raw_serial in raw_serials]
        # self.data_len = len(raw_serials[0])
        self.to_tensor = to_tensor
        self.features = features

    def reshape_data(self, raw_serial):
        stepped_serial_data = []
        for i in range(len(raw_serial) - self.time_step - self.target_mean_len):
            start = i
            end = i + self.time_step
            sequence = raw_serial[start:end]
            target = sum(raw_serial[end: end + self.target_mean_len]) / self.target_mean_len
            stepped_serial_data.append((sequence, target))
        return stepped_serial_data

    def __len__(self):
        return len(self.stepped_serial_data[0])

    def __getitem__(self, i):
        data = []
        target = []

        for raw_serial in self.stepped_serial_data:
            data_, target_ = raw_serial[i]
            data.append(list(data_))
            target.append(list([target_]))
        assert len(data) == self.features, f"{len(data)=},{self.features=}"
        if self.to_tensor:
            return torch.tensor(data), torch.tensor(target)
        else:
            return data, target


def main():
    # sin_serial = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    # , Unnamed: 0, ts_code, trade_date, open, close, high, low, pre_close, change, pct_chg, swing, vol
    features = ['open', 'close', 'high', 'low', 'change', 'pct_chg']
    price_df = pd.read_csv('../stock_fetching/SPX.csv')

    features_ = [np.array(price_df[feature]) for feature in features]
    features = [scaling(feature) for feature in features_]

    # print(features[0])
    # price = np.array(price)

    # df = pd.read_csv('../datas/airline_passengers.csv')
    # airline_passengers = df['Passengers'].tolist()
    serial_dataset = MultiFeatureDataset(features, time_step=10, target_mean_len=1, features=len(features),
                                         to_tensor=True)
    serial_dataloader = DataLoader(serial_dataset, batch_size=1, shuffle=True, num_workers=2)

    for i, (data, target) in enumerate(serial_dataloader):
        # data -> (batch, features, len)
        print(data.shape)
        break


if __name__ == '__main__':
    # main()
    pass
