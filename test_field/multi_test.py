import torch
import numpy as np

torch_list = [torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 2.]), torch.tensor([1., 3., 3.]),
              torch.tensor([2., 4., 4.])]
torch_list_ = [item.unsqueeze(0) for item in torch_list]

target_tensor = torch.concat(torch_list_, dim=0)
weights = [0.1, 0.2, 0.3, 0.4]

print()


def weighted_pooling(weights_list: list, target_tensor: torch.tensor):
    assert len(weights_list) == target_tensor.shape[0]
    assert len(target_tensor.shape) == 2
    embedding_size = target_tensor.shape[1]
    weights_tensor = torch.tensor([[weight] * embedding_size for weight in weights_list])
    return sum(weights_tensor * target_tensor)


def generate_weight(n_weights, decay_index=1.5, return_tensor=False):
    decay_rate = np.log(1 / n_weights) / (n_weights - 1)
    decay_rate *= decay_index

    x_values = np.arange(n_weights)
    y_values = np.exp(-decay_rate * x_values)

    total_sum = np.sum(y_values)
    y_values /= total_sum
    if return_tensor:
        return torch.tensor(y_values)
    else:
        return y_values


print(generate_weight(10, return_tensor=True))
print(weighted_pooling(weights, target_tensor))
