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
import torch

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
        dates = self.stepped_serial_data_date_stamp[i]

        if self.to_tensor:
            return torch.tensor(data).double(), torch.tensor(target).double(), dates
        else:
            return data, target, dates


def main():
    stock_df = pd.read_csv('../stock_fetching/SPX-10.csv')

    price = stock_df['close'].tolist()
    dates = stock_df['trade_date'].tolist()

    dates = [date_converter(str(date)) for date in dates]
    price = np.array(price)

    with open('../datas/news_dict.pickle', 'rb') as f:
        news_dict = pickle.load(f)

    serial_dataset = SingleFeatureSerialDatasetForST2(raw_serial=price,
                                                      date_stamps=dates,
                                                      time_step=3,
                                                      target_mean_len=1,
                                                      to_tensor=True,)

    serial_dataloader = DataLoader(serial_dataset, batch_size=2, shuffle=True, num_workers=2)

    for i, (data, target, corresponding_dates) in enumerate(serial_dataloader):
        # data -> (batch, len)
        print(corresponding_dates)
        news = [news_dict[date] for date in corresponding_dates]

        # news = [str(new) for new in (news_for_one_day for news_for_one_day in news)]

        assert len(news) == len(data)
        print(data.shape)
        print(corresponding_dates)
        # print(len(all_dates))
        # print(len(all_dates[0]))
        break


if __name__ == '__main__':
    main()
    pass
