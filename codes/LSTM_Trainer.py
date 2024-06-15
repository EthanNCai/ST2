# from ST2_Model import ViT1D_Model
from SingleFeatureDataset import SerialDataset
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import copy
import LSTM_Visualizer
from sklearn.metrics import mean_absolute_percentage_error
from LSTM_Model import SimpleLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16

epochs = 20
time_step = 12
learning_rate = 0.0001
# target_mean_len = 1

train_test_ratio = 0.8

lstm = SimpleLSTM(input_size=1, hidden_size=50, num_layers=8).to(device)


def scaling(raw_data):
    scaler = preprocessing.StandardScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)
    # raw_data=raw_data.reshape(-1, 1)
    # raw_data = TimeSeriesScalerMeanVariance(mu=0.,
    #                              std=1.).fit_transform(raw_data)
    # return raw_data.reshape(-1)


def main():
    # sin_train = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.1
    # sin_test = np.sin(np.arange(500) * 0.02) + np.random.randn(500) * 0.02
    # df = pd.read_csv('../datas/airline_passengers.csv')
    # airline_passengers = df['Passengers'].tolist()
    price_df = pd.read_csv('../stock_fetching/HSI-10-VOF.csv')
    vol = np.array(price_df['vol'].tolist())
    flag = np.array(price_df['is_vol_outliers'].tolist())
    # airline_passengers = np.sin(np.arange(10000) * 0.1) + np.random.randn(10000) * 0.3

    # train = airline_passengers[:int(len(airline_passengers) * train_test_ratio)]
    # test = airline_passengers[:int(len(airline_passengers) * train_test_ratio)]

    scaled_vol = scaling(vol)

    print('raw amount >>>', len(scaled_vol))
    print('train amount >>>', int(len(scaled_vol) * train_test_ratio))
    # print(len(scaled_price))
    test = scaled_vol[int(len(scaled_vol) * train_test_ratio):]
    flag_test = flag[int(len(flag) * train_test_ratio):]
    train = scaled_vol[:int(len(scaled_vol) * train_test_ratio)]
    flag_train = flag[:int(len(flag) * train_test_ratio)]

    train_serial = SerialDataset(train, time_step=time_step,
                                 to_tensor=True, flag=flag_train)
    test_serial = SerialDataset(test, time_step=time_step,
                                to_tensor=True, flag=flag_test)
    train_loader = DataLoader(train_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True)
    test_loader = DataLoader(test_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                             drop_last=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    for epoch_index in range(epochs):
        lstm.train()
        for batch_index, (data, target) in enumerate(train_loader):
            # data -> (batch, len)
            data = data.to(device).to(dtype=torch.float32)
            data = data.unsqueeze(2)
            target = target.to(device).to(dtype=torch.float32)
            output = lstm(data)
            target = target.unsqueeze(1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
            f"batch:{batch_index}/{len(train_loader)}, epoch:{epoch_index}/{epochs}, BCE_loss:{round(loss.item(), 3)}")
        # lstm.eval()
        # test_BCE_loss = 0
        # true_labels = []
        # predicted_labels = []

        # for i, (data, target) in enumerate(test_loader):
        #     with torch.no_grad():
        #         data = data.unsqueeze(2).to(device).to(dtype=torch.float32)
        #         target = target.to(device).to(dtype=torch.float32)
        #         output = lstm(data)
        #         output = output.squeeze(-1)
        #         loss_ = criterion(output, target)
        #         test_BCE_loss += loss_.item()
        #
        #         predicted = torch.round(output).cpu().numpy()
        #         true = target.cpu().numpy()
        #         predicted_labels.extend(predicted)
        #         true_labels.extend(true)
        #
        # test_loss = test_BCE_loss / len(test_loader)
        # f1 = f1_score(true_labels, predicted_labels)
        #
        # # 计算precision
        # true_positive = np.sum(np.logical_and(true_labels, predicted_labels))
        # false_positive = np.sum(np.logical_and(np.logical_not(true_labels), predicted_labels))
        # precision = true_positive / (true_positive + false_positive)
        #
        # # 计算recall
        # false_negative = np.sum(np.logical_and(true_labels, np.logical_not(predicted_labels)))
        # recall = true_positive / (true_positive + false_negative)
        #
        # # 计算accuracy
        # accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
        #
        # print(f"Test Loss: {test_loss:.4f}, F1 Score: {f1:.4f}")
        # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")


#
if __name__ == '__main__':
    main()
