from model import ST2_Model
from dataset import SerialDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 10
time_step = 256
patch_size = 16
patch_token_dim = 1024
learning_rate = 0.001

if time_step % patch_size != 0:
    print("invalid patch size ! time_step % patch_size must equal to 0")
    quit()

st2 = ST2_Model(
    seq_len=time_step,
    patch_size=patch_size,
    num_classes=1,
    channels=1,
    dim=patch_token_dim,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)


def main():
    sin_train = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    sin_test = np.sin(np.arange(500) * 0.02) + np.random.randn(500) * 0.02

    sin_train_serial_dataset = SerialDataset(sin_train, time_step=time_step, target_mean_len=1, to_tensor=True)
    sin_test_serial_dataset = SerialDataset(sin_test, time_step=time_step, target_mean_len=1, to_tensor=True)
    sin_train_dataloader = DataLoader(sin_train_serial_dataset, batch_size=batch_size, shuffle=True, num_workers=2,drop_last=True)
    sin_test_dataloader = DataLoader(sin_test_serial_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(st2.parameters(), lr=learning_rate)
    for epoch_index in range(epochs):
        for batch_index, (data, target) in enumerate(sin_train_dataloader):
            # data -> (batch, len)
            data = data.unsqueeze(1).double().to(device)
            # torch.Size([32, 1, 256])
            target = target.double().to(device)
            # torch.Size([32])

            output = st2(data, 2)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"batch:{batch_index}/{len(sin_train_dataloader)}, epoch:{epoch_index}/{epochs}, loss:{loss.item()}")


if __name__ == '__main__':
    main()