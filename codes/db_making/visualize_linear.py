from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import hashlib
import numpy as np
from datetime import datetime, timedelta
from TEU import TextExtractionUnit
import torch
from get_vec import get_vec
from tqdm import tqdm
import time
import datetime

# n_days = 15
stock_name = 'HSI-10'


def sha256_str(str_in: str) -> str:
    return hashlib.md5(str_in.encode('utf-8')).hexdigest()


def load_pickle_dict(news_dict_pickle_path):
    import pickle
    with open(news_dict_pickle_path, "rb") as f:
        news_dict_: dict = pickle.load(f)
        # print('news_dict_len >>> ', len(news_dict_.keys()))
    return news_dict_


def get_prevention_zone(start_date, end_date):
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += datetime.timedelta(days=1)
    return date_list


def similarity_search(target, candidates, top_k):
    return top_k


news_dict = load_pickle_dict('../../datas/news_dict.pickle')
vol_dict = load_pickle_dict(f'../../stock_fetching/{stock_name}-DICT.pickle')
embedding_model_path = '../../moka-ai/m3e-base'

client = PersistentClient(path='./dbs/n_days_5')
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embedding_model_path, device='mps')

collection = client.get_collection(name="default",
                                   embedding_function=embedding_function, )

device = 'mps'

list1 = []
list2 = []

all = collection.get(include=['embeddings', 'metadatas'])
all_embeddings = all['embeddings']
all_metadatas = all['metadatas']
print(all.keys())

for embedding, metadata in tqdm(zip(all_embeddings, all_metadatas)):
    x_peek_info_dict = eval(metadata['peek_info_dict'])
    x_end_info_dict = eval(metadata['end_info_dict'])
    x_pct_dict = eval(metadata['pct_dict'])
    x_start_date = metadata['start_date']
    x_end_date = metadata['end_date']
    x_peek_day = metadata['peek_day']
    prevention_zone = get_prevention_zone(x_start_date, '2024-05-01')
    try:
        y = collection.query(query_embeddings=[embedding],
                             n_results=30,
                             where=
                             {
                                 'end_date': {
                                     '$nin': prevention_zone
                                 }
                             })

        y_metadatas_list = y['metadatas'][0]
        y_metadata = y_metadatas_list[25]
        y_peek_info_dict = eval(y_metadata['peek_info_dict'])
        y_end_info_dict = eval(y_metadata['end_info_dict'])
        y_pct_dict = eval(y_metadata['pct_dict'])
        y_start_date = y_metadata['start_date']
        y_end_date = y_metadata['end_date']
        y_peek_day = y_metadata['peek_day']

        assert x_pct_dict.keys() == y_pct_dict.keys()
        list1.append(tuple(x_pct_dict[key] for key in x_pct_dict.keys()))
        list2.append(tuple(y_pct_dict[key] for key in y_pct_dict.keys()))
        #print(list1[0])#
    except:
        pass
        # print('something went wrong')

assert len(list1) == len(list2)

import matplotlib.pyplot as plt

# 创建一个包含5个子图的figure
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# 散点图(xan,yan)
axes[0, 0].scatter([x[0] for x in list1], [y[0] for y in list2])
axes[0, 0].set_title("Scatter Plot (xa, ya)")
axes[0, 0].set_xlabel("xa")
axes[0, 0].set_ylabel("ya")

# 散点图(xbn,ybn)
axes[0, 1].scatter([x[1] for x in list1], [y[1] for y in list2])
axes[0, 1].set_title("Scatter Plot (xb, yb)")
axes[0, 1].set_xlabel("xb")
axes[0, 1].set_ylabel("yb")

# 散点图(xcn,ycn)
axes[1, 0].scatter([x[2] for x in list1], [y[2] for y in list2])
axes[1, 0].set_title("Scatter Plot (xc, yc)")
axes[1, 0].set_xlabel("xc")
axes[1, 0].set_ylabel("yc")

# 散点图(xdn,ydn)
axes[1, 1].scatter([x[3] for x in list1], [y[3] for y in list2])
axes[1, 1].set_title("Scatter Plot (xd, yd)")
axes[1, 1].set_xlabel("xd")
axes[1, 1].set_ylabel("yd")

# 散点图(xen,yen)
axes[2, 0].scatter([x[4] for x in list1], [y[4] for y in list2])
axes[2, 0].set_title("Scatter Plot (xe, ye)")
axes[2, 0].set_xlabel("xe")
axes[2, 0].set_ylabel("ye")

# 调整子图间距并显示图形
plt.subplots_adjust(hspace=0.5)
plt.show()

"""
nive = {'peek_info_dict': "{'open': 22686.21, 'close': 22696.01, 'high': 22866.56, 'low': 22627.51, 'vol': 1209932.0}",
        'end_info_dict': "{'open': 22858.79, 'close': 22760.24, 'high': 22858.79, 'low': 22657.95, 'vol': 1043678.9}",
        'pct_dict': "{'open': 0.007607264501210284, 'close': 0.002830012852479498, 'high': -0.0003397975034286065, 'low': 0.0013452651219688922, 'vol': -0.1374069782434054}",
        'start_date': '2014-04-01',
        'end_date': '2014-04-16',
        'peek_day': '2014-04-17'}
"""
