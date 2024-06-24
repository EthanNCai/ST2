import pandas as pd
from datetime import datetime
import pickle
def convert_date_format(date_str):
    date_str = str(date_str)
    return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")


NAME = 'HSI-10'
sliding_var = 10
df = pd.read_csv(NAME + '.csv')
vol = df['vol']
df['trade_date'] = df['trade_date'].apply(convert_date_format)

vol_dict = {date: vol for date, vol in zip(df['trade_date'], df['vol'])}

# 打印结果以验证
with open('vol_dict.pickle', 'wb') as f:
    pickle.dump(vol_dict, f)

# print(date.head)
