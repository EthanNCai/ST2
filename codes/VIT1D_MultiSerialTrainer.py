from ST2_Model import ViT1D_Model
# from SingleSerialDataset import SerialDataset
from MultiFeatureSerialDataset import MultiFeatureDataset
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import LSTM_Visualizer
from sklearn.metrics import mean_absolute_percentage_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 1
time_step = 256
patch_size = 4
patch_token_dim = 128
learning_rate = 0.001
target_mean_len = 1
train_test_ratio = 0.9

if time_step % patch_size != 0:
    print("invalid patch size ! time_step % patch_size must equal to 0")
    quit()

vit1d = ViT1D_Model(
    seq_len=time_step,
    patch_size=patch_size,
    num_classes=1,
    channels=6,
    dim=patch_token_dim,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.05,
    emb_dropout=0.05
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
    features = ['open', 'close', 'high', 'low', 'change', 'pct_chg']
    price_df = pd.read_csv('../stock_fetching/SPX-10.csv')

    features_ = [np.array(price_df[feature]) for feature in features]
    features = [scaling(feature) for feature in features_]
    # airline_passengers = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.3

    # train = airline_passengers[:int(len(airline_passengers) * train_test_ratio)]
    # test = airline_passengers[:int(len(airline_passengers) * train_test_ratio)]

    # raw_data = scaling(price)

    print('raw amount >>>', len(features[0]))
    print('train amount >>>', int(len(features[0]) * train_test_ratio))
    # print(len(raw_data))
    test = []
    train = []
    for single_feature in features:
        test.append(single_feature[int(len(features[0]) * train_test_ratio):])
        train.append(single_feature[:int(len(features[0]) * train_test_ratio)])

    print(len(test[0]))
    print(len(train[0]))

    train_serial = MultiFeatureDataset(train, time_step=time_step,
                                       target_mean_len=target_mean_len,
                                       features=6,
                                       to_tensor=True)
    test_serial = MultiFeatureDataset(test, time_step=time_step,
                                      target_mean_len=target_mean_len,
                                      features=6,
                                      to_tensor=True)
    train_loader = DataLoader(train_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True)
    test_loader = DataLoader(test_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                             drop_last=True)

    print(f"{len(train_loader)=}")
    print(f"{len(test_loader)=}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vit1d.parameters(), lr=learning_rate)
    for epoch_index in range(epochs):
        vit1d.train()
        for batch_index, (data, target) in enumerate(train_loader):
            # data -> (batch, channel (features), len)

            data = data.to(device).to(dtype=torch.float32)

            # torch.Size([32, channel (features), 256])
            target = target.to(device).to(dtype=torch.float32)
            # torch.Size([32])

            # print(data.shape)
            # quit()

            output = vit1d(data)

            output = output.squeeze(-1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f"batch:{batch_index}/{len(train_loader)}, epoch:{epoch_index}/{epochs}, loss:{round(loss.item(), 3)}")

        # st2.eval()

        test_MSE_loss = 0
        test_MAPE_loss = 0
        for i, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                data = data.to(device).to(dtype=torch.float32)

                # torch.Size([32, channel (features), 256])
                target = target.to(device).to(dtype=torch.float32)
                output = vit1d(data)
                output = output.squeeze(-1)
                MSE_loss = criterion(output, target)
                MAPE_loss = mean_absolute_percentage_error(output.cpu(), target.cpu())
                test_MSE_loss += MSE_loss.item()
                test_MAPE_loss += MAPE_loss

        print(
            f"test --> ,MSE_loss:{round(test_MSE_loss / len(test_loader), 3)},MAPE_loss:{round(test_MAPE_loss / len(test_loader), 3)}")

    # train finished
    # Visualizer.visualizer(train, test, 5, time_step, st2, device=device)


if __name__ == '__main__':
    main()
