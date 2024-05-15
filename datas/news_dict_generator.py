import pandas as pd
import pickle
from datetime import datetime

from typing import Dict

df = pd.read_csv('1640337222.csv')


df = df[['微博正文', '发布时间']]

news_dict: Dict[str,list] = {}

for index, row in df.iterrows():
    time = row['发布时间']
    news_text = row['微博正文']
    time_obj = datetime.strptime(time, "%Y-%m-%d %H:%M")
    formatted_time = time_obj.strftime("%Y-%m-%d")
    row['发布时间'] = formatted_time

    if formatted_time not in news_dict:
        news_dict[formatted_time] = [news_text]
    else:
        news_dict[formatted_time].append(news_text)


# df = pd.DataFrame(columns=[f'v{i}' for i in range(768)])
#
#
# for i in range(7):
#     vec = np.random.rand(768)
#     df.loc[i] = vec

# print(time_to_news_dict['2024-03-31'])

with open('news_dict.pickle', 'wb') as f:
    pickle.dump(news_dict, f)
