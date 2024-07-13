import pandas as pd
from datetime import datetime
import pickle
import random


def convert_date_format(date_str):
    date_str = str(date_str)
    return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")


NAME = 'HSI-10'
sliding_var = 10
df = pd.read_csv(NAME + '.csv')
vol = df['vol']
df['trade_date'] = df['trade_date'].apply(convert_date_format)

stock_dict = {date: {'open': open_, 'close': close, 'high': high, 'low': low, 'vol': vol} for
              date, open_, close, high, low, vol in
              zip(df['trade_date'], df['open'], df['close'], df['high'], df['low'], df['vol'])}

rand_data = stock_dict[random.choice(list(stock_dict.keys()))]
print(rand_data)
with open(NAME + '-DICT.pickle', 'wb') as f:
    pickle.dump(stock_dict, f)
print('done')
