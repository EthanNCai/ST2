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
    def __init__(self, raw_serial, time_step, target_mean_len, to_tensor=True):
        self.time_step = time_step
        self.target_mean_len = target_mean_len
        self.stepped_serial_data = self.reshape_data(raw_serial)
        self.to_tensor = to_tensor

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
        return len(self.stepped_serial_data)

    def __getitem__(self, i):
        data, target = self.stepped_serial_data[i]
        #data += target

        # label = 1 if target >= now else 0
        if self.to_tensor:
            return torch.tensor(data).double(), torch.tensor(target).double()
        else:
            return data, target


def main():
    # sin_serial = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    df = pd.read_csv('../datas/airline_passengers.csv')
    airline_passengers = df['Passengers'].tolist()
    serial_dataset = SerialDataset(airline_passengers, time_step=64, target_mean_len=1, to_tensor=True)
    serial_dataloader = DataLoader(serial_dataset, batch_size=1, shuffle=True, num_workers=2)

    for i, (data, target) in enumerate(serial_dataloader):
        # data -> (batch, len)
        print(data.shape)
        break



# main()
