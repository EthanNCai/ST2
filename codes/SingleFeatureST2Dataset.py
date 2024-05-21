import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
import pickle
import collections
import contextlib
import re
from ST2_Model import ST2
import torch
from TEU import TextExtractionUnit

def load_the_news(batch_size,patch_size, corresponding_dates, news_dict):

    batched_news = []
    for _ in range(batch_size):
        batched_news.append([])
    for corresponding_date in corresponding_dates:
        for b in range(batch_size):
            if False and corresponding_date[b] in news_dict:
                batched_news[b].extend(news_dict[corresponding_date[b]])
            else:
                batched_news[b].extend([' '])
    # patching the batched news
    patched_news = []
    for news_batch in batched_news:
        patched_news.append([news_batch[i:i + patch_size] for i in range(0, len(news_batch), patch_size)])
    return patched_news


def get_news(news):
    return news


def scaling(raw_data):
    scaler = preprocessing.MinMaxScaler()
    raw_data = scaler.fit_transform(np.array(raw_data).reshape(-1, 1))
    return raw_data.reshape(-1)


def date_converter(raw_date):
    date_obj = datetime.strptime(raw_date, "%Y%m%d")
    return date_obj.strftime("%Y-%m-%d")


class SingleFeatureSerialDatasetForST2(Dataset):
    def __init__(self, raw_serial, date_stamps, time_step, target_mean_len, to_tensor=True):
        self.time_step = time_step
        self.target_mean_len = target_mean_len
        self.stepped_serial_data = self.reshape_data(raw_serial)
        self.stepped_serial_data_date_stamp = self.reshape_data_date_stamp(date_stamps)
        self.to_tensor = to_tensor

        assert len(self.stepped_serial_data_date_stamp) == len(self.stepped_serial_data)

    def reshape_data(self, raw_serial):
        stepped_serial_data = []
        for i in range(len(raw_serial) - self.time_step - self.target_mean_len):
            start = i
            end = i + self.time_step
            sequence = raw_serial[start:end]
            target = sum(raw_serial[end: end + self.target_mean_len]) / self.target_mean_len
            stepped_serial_data.append((sequence, target))
        return stepped_serial_data

    def reshape_data_date_stamp(self, raw_serial):
        stepped_serial_data_date_stamp = []
        for i in range(len(raw_serial) - self.time_step - self.target_mean_len):
            start = i
            end = i + self.time_step
            sequence = raw_serial[start:end]
            stepped_serial_data_date_stamp.append(sequence)
        return stepped_serial_data_date_stamp

    def __len__(self):
        return len(self.stepped_serial_data)

    def __getitem__(self, i):
        data, target = self.stepped_serial_data[i]

        now =  data[-1]

        label = 1 if target >= now else 0

        dates = self.stepped_serial_data_date_stamp[i]

        # print('get_item >>>',dates)
        if self.to_tensor:
            return torch.tensor(data).double(), torch.tensor(label).double(), dates
        else:
            return data, label, dates


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 3
    epochs = 4
    time_step = 4
    patch_size = 2
    patch_token_dim = 1024
    mlp_dim = 64
    learning_rate = 0.001
    target_mean_len = 1
    train_test_ratio = 0.8
    dropout = 0.1

    stock_df = pd.read_csv('../stock_fetching/SPX-10.csv')

    price = stock_df['close'].tolist()
    dates = stock_df['trade_date'].tolist()

    dates = [date_converter(str(date)) for date in dates]
    price = np.array(price)

    with open('../datas/news_dict.pickle', 'rb') as f:
        news_dict = pickle.load(f)
    teu = TextExtractionUnit('/home/cjz/models/bert-base-chinese/', dim_input=768, dim_output=1024).to(device)
    # teu = TextExtractionUnit('../google-bert/bert-base-chinese/', dim_input=768, dim_output=1024).to(device)

    serial_dataset = SingleFeatureSerialDatasetForST2(raw_serial=price,
                                                      date_stamps=dates,
                                                      time_step=time_step,
                                                      target_mean_len=1,
                                                      to_tensor=True, )

    serial_dataloader = DataLoader(serial_dataset, batch_size=batch_size, shuffle=True, num_workers=2)



    # st2 = ST2(
    #     seq_len=time_step,
    #     patch_size=patch_size,
    #     num_classes=1,
    #     channels=1,
    #     dim=patch_token_dim,
    #     depth=8,
    #     heads=8,
    #     mlp_dim=mlp_dim,
    #     dropout=dropout,
    #     emb_dropout=dropout
    # ).to(device)

    for i, (data, target, corresponding_dates) in enumerate(serial_dataloader):
        # data -> (B, L)  len is actually
        # convert (B, L) -> (B, C, L)
        data = data.unsqueeze(1)
        # print(data.shape)

        # batching the news
        batched_news = []
        for _ in range(batch_size):
            batched_news.append([])
        for corresponding_date in corresponding_dates:
            for b in range(batch_size):
                if False and corresponding_date[b] in news_dict:
                    batched_news[b].extend(news_dict[corresponding_date[b]])
                else:
                    batched_news[b].extend([' '])
        # patching the batched news
        patched_news = []
        for news_batch in batched_news:
            patched_news.append([news_batch[i:i + patch_size] for i in range(0, len(news_batch), patch_size)])

        # desired ->
        print("batched_news >>> ", batched_news)
        print("patched_news >>> ", patched_news)

        # batched_news >> > [[' ', ' ', ' ', ' '], [' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ']]
        # patched_news >> > [[[' ', ' '], [' ', ' ']], [[' ', ' '], [' ', ' ']], [[' ', ' '], [' ', ' ']]]

        news_embeddings_list = []
        for news_batch in patched_news:
            news_embeddings_list.append(torch.concat([teu(news_patch) for news_patch in news_batch], dim=0))
        news_embeddings_tensor = torch.concat([patch_embedding for patch_embedding in news_embeddings_list], dim=0)
        news_embeddings_tensor = news_embeddings_tensor.view(batch_size,time_step//patch_size, -1)
        # print(news_embeddings_tensor.shape)

        # desired shape ==> [B,P,L]
        # print(text_embeddings)
        """
        HERE !!!! 2024.5.16 night
        """

        # output = st2(data, news_embeddings)

        assert len(batched_news) == batch_size
        break


if __name__ == '__main__':
    main()
    pass
