import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn import preprocessing


def scaling(raw_data):
    scaler = preprocessing.StandardScaler()
    raw_data = copy.deepcopy(raw_data)
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


class SerialDataset(Dataset):
    def __init__(self, raw_serial, flag, time_step, to_tensor=True):
        self.time_step = time_step
        self.flag = flag
        self.stepped_serial_data = self.reshape_data(raw_serial)
        self.to_tensor = to_tensor

    def reshape_data(self, raw_serial):
        stepped_serial_data = []
        for i in range(len(raw_serial) - self.time_step):
            start = i
            end = i + self.time_step
            sequence = raw_serial[start:end]
            target = self.flag[end]
            stepped_serial_data.append((sequence, target))
        return stepped_serial_data

    def __len__(self):
        return len(self.stepped_serial_data)

    def __getitem__(self, i):
        data, target = self.stepped_serial_data[i]
        # data += target

        # label = 1 if target >= now else 0
        if self.to_tensor:
            return torch.tensor(data).double(), torch.tensor(target).double()
        else:
            return data, target


def main():
    # sin_serial = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    price_df = pd.read_csv('../stock_fetching/HSI-10-VOF.csv')
    price = np.array(price_df['vol'].tolist())
    flag = np.array(price_df['is_vol_outliers'].tolist())

    # train_test_ratio = 0.7
    batch_size = 5
    time_step = 3

    dataset = SerialDataset(price, time_step=time_step,
                            to_tensor=True, flag=flag)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                            drop_last=True)

    for i, (data, target) in enumerate(dataloader):
        # data -> (batch, len)
        print(data.shape)
        print(target.shape)
        print(target)
        break


# if __name__ == '__main__':
    # main()
