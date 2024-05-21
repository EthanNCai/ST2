from ST2_Model import ST2
from SingleFeatureST2Dataset import SingleFeatureSerialDatasetForST2, date_converter, load_the_news
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_percentage_error
import os

from tslearn.preprocessing import TimeSeriesScalerMeanVariance

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
epochs = 100
time_step = 32
patch_size = 2
token_dim = 128
mlp_dim = 256
learning_rate = 0.001
target_mean_len = 1
train_test_ratio = 0.8
dropout = 0.1
alpha = 0.5
teu_dropout = 0.15
pooling_mode = "max"
stock = 'SPX'
shuffle_train = True

# pooling_mode = "max"

if time_step % patch_size != 0:
    print("invalid patch size ! time_step % patch_size must equal to 0")
    quit()

st2 = ST2(
    seq_len=time_step,
    patch_size=patch_size,
    num_classes=1,
    channels=1,
    dim=token_dim,
    depth=8,
    heads=8,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=dropout,
    text_emb_model_path='/home/cjz/models/bert-base-chinese/',
    token_dim=token_dim,
    alpha=alpha,
    teu_dropout=teu_dropout,
    pooling_mode=pooling_mode,
).to(device)



def scaling(raw_data):
    return TimeSeriesScalerMeanVariance(mu=0.,
                                 std=1.).fit_transform(raw_data)


def main():
    print("Semantic Time Series Transformer (ST2)")
    print("ST2 Configurations >>>>>>>>>>>>>>>>>>>>>>>>")
    print(f"  batch_size: {batch_size}")
    print(f"  epochs: {epochs}")
    print(f"  time_step: {time_step}")
    print(f"  patch_size: {patch_size}")
    print(f"  token_dim: {token_dim}")
    print(f"  mlp_dim: {mlp_dim}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  target_mean_len: {target_mean_len}")
    print(f"  train_test_ratio: {train_test_ratio}")
    print(f"  dropout: {dropout}")
    print(f"  alpha: {alpha}")
    print(f"  teu_dropout: {teu_dropout}")
    print(f"  pooling_mode: {pooling_mode}")
    print(f"  shuffle_train: {shuffle_train}")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    stock_df = pd.read_csv(f'../stock_fetching/{stock}-10.csv')

    price = stock_df['close'].tolist()
    dates = stock_df['trade_date'].tolist()

    dates = [date_converter(str(date)) for date in dates]
    price = np.array(price)

    with open('../datas/news_dict.pickle', 'rb') as f:
        news_dict = pickle.load(f)

    raw_data = price

    print('raw amount >>>', len(raw_data))
    print('train amount >>>', int(len(raw_data) * train_test_ratio))
    # print(len(raw_data))
    test = raw_data[int(len(raw_data) * train_test_ratio):]
    train = raw_data[:int(len(raw_data) * train_test_ratio)]
    dates_test = dates[int(len(dates) * train_test_ratio):]
    dates_train = dates[:int(len(dates) * train_test_ratio)]

    train_serial = SingleFeatureSerialDatasetForST2(train, time_step=time_step,
                                                    target_mean_len=target_mean_len,
                                                    date_stamps=dates_train,
                                                    to_tensor=True)
    test_serial = SingleFeatureSerialDatasetForST2(test, time_step=time_step,
                                                   target_mean_len=target_mean_len,
                                                   date_stamps=dates_test,
                                                   to_tensor=True)
    train_loader = DataLoader(train_serial, batch_size=batch_size, shuffle=shuffle_train, num_workers=2,
                              drop_last=True)
    test_loader = DataLoader(test_serial, batch_size=batch_size, shuffle=True, num_workers=2,
                             drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(st2.parameters(), lr=learning_rate)
    for epoch_index in range(epochs):
        st2.train()
        for i, (data, target, corresponding_dates) in enumerate(train_loader):
            # data -> (batch, len)


            data = data.unsqueeze(1).to(device).to(dtype=torch.float32)

            # torch.Size([32, 1, 256])
            target = target.to(device).to(dtype=torch.float32)
            # torch.Size([32])

            batched_news = load_the_news(batch_size,patch_size,corresponding_dates,news_dict)

            output = st2(data,batched_news)

            output = output.squeeze(-1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(
                f"batch:{i}/{len(train_loader)}, epoch:{epoch_index}/{epochs}, loss:{round(loss.item(), 3)}")

        st2.eval()
        # with torch.no_grad():
        test_MSE_loss = 0
        test_MAPE_loss = 0
        for i, (data, target, corresponding_dates) in enumerate(test_loader):
            # data -> (batch, len)
            with torch.no_grad():

                data = data.unsqueeze(1).to(device).to(dtype=torch.float32)
                # torch.Size([32, 1, 256])
                target = target.to(device).to(dtype=torch.float32)
                # torch.Size([32])

                batched_news = load_the_news(batch_size, patch_size, corresponding_dates, news_dict)
                output = st2(data, batched_news)

                output = output.squeeze(-1)
                MSE_loss = criterion(output, target)
                MAPE_loss = mean_absolute_percentage_error(output.cpu(), target.cpu())
                test_MSE_loss += MSE_loss.item()
                test_MAPE_loss += MAPE_loss
                # print(f"batch -> ({i}/{len(test_loader)})")

        torch.save(st2, f'../checkpoints/st2_mape_{round(test_MAPE_loss / len(test_loader), 3)}_{stock}_.pth')
        print(f"evaluate_set --> MSE_loss:{round(test_MSE_loss / len(test_loader), 3)}, MAPE_loss:{round(test_MAPE_loss / len(test_loader), 3)}({round((test_MAPE_loss / len(test_loader)) * 100, 2)}%)")
        # print("model_saved")
        # train finished
    # Visualizer.visualizer(train, test, 5, time_step, st2, device=device)


if __name__ == '__main__':
    main()
