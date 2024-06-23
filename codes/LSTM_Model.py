from torch import nn
import torch


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=18, hidden_size=50, num_layers=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(50, 25)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(25, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


def test():
    data = torch.rand((32, 9, 18))
    model = SimpleLSTM()
    print(data.shape)
    output = model(data)
    output = output.squeeze(-1)
    print(output.shape)



# test()