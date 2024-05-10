import matplotlib.pyplot as plt
import torch
import copy
import numpy as np


def visualizer(datas: np.ndarray, predict_steps, time_step, model, *, device):
    assert isinstance(datas, np.ndarray)
    assert np.ndim(datas) == 1
    data_pred = copy.deepcopy(datas).tolist()  # here

    for _ in range(predict_steps):
        # print(data_pred[-time_step:])
        data = torch.tensor(data_pred[-time_step:]).to(device).to(dtype=torch.float32)
        data = data.unsqueeze(0).unsqueeze(0)
        assert data.shape == torch.Size([1, 1, time_step])
        output = model(data)
        output = output.squeeze(-1)
        output = output.detach().tolist()
        data_pred += output  # here

    plt.plot(data_pred, c='r')
    plt.plot(datas, c='g')
    plt.show()
