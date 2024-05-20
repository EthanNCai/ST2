# from ST2_Model import ViT1D_Model
from SingleFeatureDataset import SerialDataset
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import Visualizer
from sklearn.metrics import mean_absolute_percentage_error
from LSTM_Model import SimpleLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
epochs = 100
time_step = 32
learning_rate = 0.001
target_mean_len = 1
train_test_ratio = 0.8

lstm = SimpleLSTM(input_size=time_step, hidden_size=50, num_layers=8, ).to(device)


def main():
    # sin_train = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    # sin_test = np.sin(np.arange(500) * 0.02) + np.random.randn(500) * 0.02
    # df = pd.read_csv('../datas/airline_passengers.csv')
    # airline_passengers = df['Passengers'].tolist()
    price_df = pd.read_csv('../stock_fetching/SPX-10.csv')
    price = price_df['close'].tolist()
    price = np.array(list(reversed(price)))
    # airline_passengers = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.3

    # train = airline_passengers[:int(len(airline_passengers) * train_test_ratio)]
    # test = airline_passengers[:int(len(airline_passengers) * train_test_ratio)]

    raw_data = price
    scaler = preprocessing.MinMaxScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    raw_data = raw_data.reshape(-1)

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
    train_loader = DataLoader(train_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True)
    test_loader = DataLoader(test_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                             drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    for epoch_index in range(epochs):
        lstm.train()
        for batch_index, (data, target) in enumerate(train_loader):
            # data -> (batch, len)

            data = data.unsqueeze(1).to(device).to(dtype=torch.float32)

            # torch.Size([32, 1, 256])
            target = target.to(device).to(dtype=torch.float32)
            # torch.Size([32])

            output = lstm(data)

            output = output.squeeze(-1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(
            #     f"batch:{batch_index}/{len(train_loader)}, epoch:{epoch_index}/{epochs}, loss:{round(loss.item(), 3)}")

        lstm.eval()

        test_MSE_loss = 0
        test_MAPE_loss = 0

        for i, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                data = data.unsqueeze(1).to(device).to(dtype=torch.float32)
                target = target.to(device).to(dtype=torch.float32)
                output = lstm(data)
                output = output.squeeze(-1)
                MSE_loss = criterion(output, target)
                MAPE_loss = mean_absolute_percentage_error(output.cpu(), target.cpu())
                test_MSE_loss += MSE_loss.item()
                test_MAPE_loss += MAPE_loss

        print(f"test --> MSE_loss:{round(test_MSE_loss / len(test_loader), 3)},MAPE_loss:{round(test_MAPE_loss / len(test_loader), 3)}({round((test_MAPE_loss / len(test_loader))*100,2)}%)")

    # train finished

    # Visualizer.visualizer(train, test, len(test), time_step, lstm, device=device)


#
if __name__ == '__main__':
    main()
