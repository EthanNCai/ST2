from torch import nn
import torch


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(50, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.linear(x)
        x = self.softmax(x)
        return x


def test():
    # (batch_size, sequence_len, feature_size)
    data = torch.rand((3, 18, 1))
    model = SimpleLSTM()
    print(data.shape)
    output = model(data)
    print(output.shape)

    # data = torch.rand((3, 17, 1))
    # output = model(data)
    # print(output.shape)


test()
