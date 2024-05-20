from VIT1D_Model import VIT1D_Model
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 100
time_step = 64
patch_size = 4
patch_token_dim = 128
mlp_dim = 32
learning_rate = 0.001
target_mean_len = 5
train_test_ratio = 0.8
dropout = 0.1
# teu_dropout = 0.1

if time_step % patch_size != 0:
    print("invalid patch size ! time_step % patch_size must equal to 0")
    quit()

vit1d = VIT1D_Model(
    seq_len=time_step,
    patch_size=patch_size,
    num_classes=1,
    channels=1,
    dim=patch_token_dim,
    depth=8,
    heads=8,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=dropout
).to(device)


def scaling(raw_data):
    scaler = preprocessing.MinMaxScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)

def read_stock_np(path):
    price_df = pd.read_csv(path)
    price = price_df['close'].tolist()
    return np.array(price)

def main():
    spx_price = read_stock_np('../stock_fetching/SPX-10.csv')
    # nasdaq_price = read_stock_np('../stock_fetching/IXIC-10.csv')
    # dji_price = read_stock_np('../stock_fetching/DJI-10.csv')
    # hsi_price = read_stock_np('../stock_fetching/HSI-10.csv')


    spx_raw_data = scaling(spx_price)
    # nasdaq_raw_data = scaling(nasdaq_price)
    # dji_raw_data = scaling(dji_price)
    # hsi_raw_data = scaling(hsi_price)

    # print(len(raw_data))
    spx_test = spx_raw_data[int(len(spx_raw_data) * train_test_ratio):]
    spx_train = spx_raw_data[:int(len(spx_raw_data) * train_test_ratio)]
    # train = np.concatenate((spx_train, nasdaq_raw_data, dji_raw_data, hsi_raw_data))

    print('test amount >>>', len(spx_test))
    print('train amount >>>', len(spx_train))

    train_serial = SerialDataset(spx_train, time_step=time_step,
                                 target_mean_len=target_mean_len,
                                 to_tensor=True)
    test_serial = SerialDataset(spx_test, time_step=time_step,
                                target_mean_len=target_mean_len,
                                to_tensor=True)
    train_loader = DataLoader(train_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True)
    test_loader = DataLoader(test_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                             drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vit1d.parameters(), lr=learning_rate)

    # print('train_loader[0]',train_loader[0])

    for epoch_index in range(epochs):
        vit1d.train()
        for batch_index, (data, target) in enumerate(train_loader):
            # print(data.shape)
            # data -> (batch, len)
            # print(data.shape)
            data = data.unsqueeze(1).to(device).to(dtype=torch.float32)

            # torch.Size([32, 1, 256])
            target = target.to(device).to(dtype=torch.float32)
            # torch.Size([32])

            output = vit1d(data)

            output = output.squeeze(-1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(
            #     f"batch:{batch_index}/{len(train_loader)}, epoch:{epoch_index}/{epochs}, loss:{round(loss.item(), 3)}")

        # st2.eval()

        test_MSE_loss = 0
        test_MAPE_loss = 0
        for i, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                data = data.unsqueeze(1).to(device).to(dtype=torch.float32)
                target = target.to(device).to(dtype=torch.float32)
                output = vit1d(data)
                output = output.squeeze(-1)
                MSE_loss = criterion(output, target)
                MAPE_loss = mean_absolute_percentage_error(output.cpu(), target.cpu())
                test_MSE_loss += MSE_loss.item()
                test_MAPE_loss += MAPE_loss

        print(
            f"evaluate_set --> MSE_loss:{round(test_MSE_loss / len(test_loader), 3)}, MAPE_loss:{round(test_MAPE_loss / len(test_loader), 3)}({round((test_MAPE_loss / len(test_loader)) * 100, 2)}%)")
    # train finished
    # Visualizer.visualizer(train, test, 5, time_step, st2, device=device)


if __name__ == '__main__':
    main()
