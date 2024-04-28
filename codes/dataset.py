import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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
        if self.to_tensor:
            return torch.tensor(data), torch.tensor(target)
        else:
            return data, target


def main():
    sin_serial = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    serial_dataset = SerialDataset(sin_serial, time_step=50, target_mean_len=1,to_tensor=True)
    serial_dataloader = DataLoader(serial_dataset, batch_size=1, shuffle=True, num_workers=2)

    for i, (data, target) in enumerate(serial_dataloader):
        break


if __name__ == '__main__':
    main()
