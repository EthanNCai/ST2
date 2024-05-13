from ST2_Model import ST2_Model
from SingleSerialDataset import SerialDataset
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import Visualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 1
time_step = 256
patch_size = 2
patch_token_dim = 128
learning_rate = 0.001
target_mean_len = 5
train_test_ratio = 0.9

if time_step % patch_size != 0:
    print("invalid patch size ! time_step % patch_size must equal to 0")
    quit()

st2 = ST2_Model(
    seq_len=time_step,
    patch_size=patch_size,
    num_classes=1,
    channels=1,
    dim=patch_token_dim,
    depth=8,
    heads=8,
    mlp_dim=64,
    dropout=0.1,
    emb_dropout=0.1
).to(device)


def scaling(raw_data):
    scaler = preprocessing.MinMaxScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


def main():
    # sin_train = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    # sin_test = np.sin(np.arange(500) * 0.02) + np.random.randn(500) * 0.02
    # df = pd.read_csv('../datas/airline_passengers.csv')
    # airline_passengers = df['Passengers'].tolist()
    price_df = pd.read_csv('../stock_fetching/SPX.csv')
    price = price_df['close'].tolist()
    price = np.array(price)
    # airline_passengers = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.3

    # train = airline_passengers[:int(len(airline_passengers) * train_test_ratio)]
    # test = airline_passengers[:int(len(airline_passengers) * train_test_ratio)]

    raw_data = scaling(price)

    print('raw amount >>>', len(raw_data))
    print('train amount >>>', int(len(raw_data) * train_test_ratio))
    # print(len(raw_data))
    test = raw_data[int(len(raw_data) * train_test_ratio):]
    train = raw_data[:int(len(raw_data) * train_test_ratio)]

    train_serial = SerialDataset(train, time_step=time_step,
                                 target_mean_len=target_mean_len,
                                 to_tensor=True)
    test_serial = SerialDataset(test, time_step=time_step,
                                target_mean_len=target_mean_len,
                                to_tensor=True)
    train_loader = DataLoader(train_serial, batch_size=batch_size, shuffle=False, num_workers=2,
                              drop_last=True)
    test_loader = DataLoader(test_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                             drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(st2.parameters(), lr=learning_rate)
    for epoch_index in range(epochs):
        st2.train()
        for batch_index, (data, target) in enumerate(train_loader):
            # data -> (batch, len)

            data = data.unsqueeze(1).to(device).to(dtype=torch.float32)

            # torch.Size([32, 1, 256])
            target = target.to(device).to(dtype=torch.float32)
            # torch.Size([32])

            output = st2(data)

            output = output.squeeze(-1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f"batch:{batch_index}/{len(train_loader)}, epoch:{epoch_index}/{epochs}, loss:{round(loss.item(), 3)}")

        # st2.eval()

        test_loss = 0

        for i, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                data = data.unsqueeze(1).to(device).to(dtype=torch.float32)
                target = target.to(device).to(dtype=torch.float32)
                output = st2(data)
                output = output.squeeze(-1)
                loss = criterion(output, target)

                test_loss += loss.item()

        print(f"test --> ,loss:{round(test_loss / len(test_loader), 3)}")

    # train finished
    # Visualizer.visualizer(train, test, 5, time_step, st2, device=device)


if __name__ == '__main__':
    main()
