import matplotlib.pyplot as plt
import torch
import copy
import numpy as np


def visualizer(train: np.ndarray, test: np.ndarray, predict_steps, time_step, model, *, device):
    # assert isinstance(train, np.ndarray)
    # assert np.ndim(train) == 1
    # assert isinstance(test, np.ndarray)
    # assert np.ndim(test) == 1
    train = copy.deepcopy(train).tolist()  # here
    test = copy.deepcopy(test).tolist()

    all = train + test

    for _ in range(predict_steps):
        # print(train[-time_step:])
        data = torch.tensor(train[-time_step:]).to(device).to(dtype=torch.float32)
        data = data.unsqueeze(0)
        assert data.shape == torch.Size([1, 1, time_step])
        output = model(data)
        output = output.squeeze(-1)
        output = output.detach().tolist()
        train += output  # here

    # train = train[len(train):]
    plt.plot(all, c='g')
    plt.plot(train, c='r', linewidth=4)
    # plt.plot(all, c='g')
    plt.show()
