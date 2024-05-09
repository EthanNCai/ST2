from ST2_Model import ST2_Model
from Dataset import SerialDataset
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
epochs = 1
time_step = 512
patch_size = 16
patch_token_dim = 512
learning_rate = 0.001
target_mean_len = 1

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
    # sin_train = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    # sin_test = np.sin(np.arange(500) * 0.02) + np.random.randn(500) * 0.02
    df = pd.read_csv('../datas/airline_passengers.csv')
    # airline_passengers = df['Passengers'].tolist()
    airline_passengers = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.01

    # scaler = preprocessing.MinMaxScaler()
    # airline_passengers = scaler.fit_transform(np.array(airline_passengers).reshape(-1, 1))
    # airline_passengers = airline_passengers.reshape(-1)

    airline_passengers_test = airline_passengers[int(len(airline_passengers) * 0.7):]
    airline_passengers_train = airline_passengers[:int(len(airline_passengers) * 0.7)]

    sin_train_serial_dataset = SerialDataset(airline_passengers_train, time_step=time_step,
                                             target_mean_len=target_mean_len,
                                             to_tensor=True)
    sin_test_serial_dataset = SerialDataset(airline_passengers_test, time_step=time_step,
                                            target_mean_len=target_mean_len,
                                            to_tensor=True)
    sin_train_dataloader = DataLoader(sin_train_serial_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                      drop_last=True)
    sin_test_dataloader = DataLoader(sin_test_serial_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                     drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(st2.parameters(), lr=learning_rate)
    for epoch_index in range(epochs):
        st2.train()
        for batch_index, (data, target) in enumerate(sin_train_dataloader):
            # data -> (batch, len)

            data = data.unsqueeze(1).to(device).to(dtype=torch.float32)

            # torch.Size([32, 1, 256])
            target = target.to(device).to(dtype=torch.float32)
            # torch.Size([32])

            output = st2(data, 2)

            output = output.squeeze(-1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f"batch:{batch_index}/{len(sin_train_dataloader)}, epoch:{epoch_index}/{epochs}, loss:{round(loss.item(), 3)}")

        st2.eval()

        test_loss = 0

        for i, (data, target) in enumerate(sin_test_dataloader):
            with torch.no_grad():
                data = data.unsqueeze(1).to(device).to(dtype=torch.float32)
                target = target.to(device).to(dtype=torch.float32)
                output = st2(data, 2)
                output = output.squeeze(-1)
                loss = criterion(output, target)

                test_loss += loss.item()

        print(f"test --> ,loss:{round(test_loss / len(sin_test_dataloader), 3)}")

    # train finished

    airline_passengers_pred = copy.deepcopy(airline_passengers).tolist()  # here

    plt.plot(airline_passengers_pred, c='r')
    plt.plot(airline_passengers, c='g')
    plt.show()


if __name__ == '__main__':
    main()
