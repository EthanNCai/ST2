import matplotlib.pyplot as plt
import torch
import copy
import numpy as np
from dataset import SerialDataset
from torch.utils.data import DataLoader, Dataset

def visualizer(train: np.ndarray, test: np.ndarray, time_step, model, *, device):
    # assert isinstance(train, np.ndarray)
    # assert np.ndim(train) == 1
    # assert isinstance(test, np.ndarray)
    # assert np.ndim(test) == 1
    train = copy.deepcopy(train).tolist()  # here
    test = copy.deepcopy(test).tolist()

    all = train + test

    test_serial = SerialDataset(test, time_step=time_step,
                                target_mean_len=1,
                                to_tensor=True)
    test_loader = DataLoader(test_serial, batch_size=1, shuffle=False, num_workers=2,
                              drop_last=True)

    y_hat = copy.deepcopy(train)
    y = train

    for data, target in test_loader:
        with torch.no_grad():
            data = data.unsqueeze(1).to(device).to(dtype=torch.float32)
            target = target.to(device).to(dtype=torch.float32)
            output = model(data)

            output = output.detach()
            output = float(output[0])
            target = float(target[0])

            y_hat.append(output)
            y.append(target)

    plt.plot(y[int(len(test) * 1.5):], c='r', linewidth=0.6)
    plt.plot(y_hat[int(len(test) * 1.5):], c='g', linewidth=0.6)

    plt.show()
