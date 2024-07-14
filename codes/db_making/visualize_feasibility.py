from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import hashlib
from datetime import datetime, timedelta
from tqdm import tqdm
import datetime
import numpy as np
from sklearn import preprocessing

# n_days = 15
stock_name = 'HSI-10'
top_k_1 = 30
top_k_2 = 3
stage1_window = 5
stage2_window = 3

assert stage1_window > stage2_window


def sort_time_list(date_list: list):
    parsed_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in date_list]
    sorted_dates = sorted(parsed_dates)
    sorted_date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]
    return sorted_date_strings


def get_exchange_days(stock_dict: dict):
    date_list = list(stock_dict.keys())
    sorted_time_list = sort_time_list(date_list)
    return sorted_time_list


def standard_scaled_manhattan_distance(vec1, vec2):
    def scale(time_series):
        scaler = preprocessing.StandardScaler()
        time_series = scaler.fit_transform(np.array(time_series).reshape(-1, 1))
        time_series = time_series.reshape(-1)
        return time_series

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sum(np.abs(scale(vec1) - scale(vec2)))


def load_pickle_dict(news_dict_pickle_path):
    import pickle
    with open(news_dict_pickle_path, "rb") as f:
        news_dict_: dict = pickle.load(f)
        # print('news_dict_len >>> ', len(news_dict_.keys()))
    return news_dict_


stock_info_dict = load_pickle_dict(f'../../stock_fetching/{stock_name}-DICT.pickle')
exchange_days_ordered = get_exchange_days(stock_info_dict)


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


def get_xk_days_before(date_in, k):
    date = datetime.datetime.strptime(date_in, '%Y-%m-%d')
    dates = [date - datetime.timedelta(days=i) for i in range(k - 1, -1, -1)]
    dates_str = [date.strftime('%Y-%m-%d') for date in dates]
    return dates_str


def get_xk_trade_days_before(date_in, k):
    start_date_index = exchange_days_ordered.index(date_in)
    end_date_index = start_date_index + k
    dates_list = exchange_days_ordered[start_date_index:end_date_index]
    return dates_list


def date_to_day_wise_info_list(date_list: list, info_dicts):
    dw_info_dict = {}
    '''
    dw_info_dict should looks something like:  
    {
        'close' : [c1,c2,c3.....,ck2],
        'vol' : [v1,v2,v3.....,vk2]
    }
    '''
    stock_info_list = [info_dicts[date] for date in date_list]
    stock_info_keys = stock_info_list[0].keys()
    for key in stock_info_keys:
        dw_info_dict[key] = [stock_info_list[i][key] for i in range(stage2_window)]

    return dw_info_dict


def stock_info_dict_similarity(dict_1: dict, dict_2: dict):
    keys = dict_2.keys()
    similarities = [standard_scaled_manhattan_distance(dict_1[key], dict_2[key]) for key in keys]
    avg_sim = sum(similarities) / len(similarities)
    return avg_sim


def find_top_k_2(x_metadata: dict, y_metadatas: list, top_k_2: int):
    x_date_row = get_xk_days_before(x_metadata['end_date'], stage2_window)
    y_date_row_list = [get_xk_trade_days_before(metadata['end_date'], stage2_window) for metadata in y_metadatas]
    x_dw_info_row = date_to_day_wise_info_list(x_date_row, stock_info_dict)
    y_dw_info_row_list = [date_to_day_wise_info_list(date_row, stock_info_dict) for date_row in y_date_row_list]
    scores = [stock_info_dict_similarity(x_dw_info_row, y_dw) for y_dw in y_dw_info_row_list]

    print()
    return scores


device = 'cuda'

news_dict = load_pickle_dict('../../datas/news_dict.pickle')
vol_dict = load_pickle_dict(f'../../stock_fetching/{stock_name}-DICT.pickle')
embedding_model_path = '../../moka-ai/m3e-large'

client = PersistentClient(path='./dbs/n_days_5')
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embedding_model_path, device=device)

collection = client.get_collection(name="default",
                                   embedding_function=embedding_function, )

list1 = []
list2 = []

all = collection.get(include=['embeddings', 'metadatas'])
all_embeddings = all['embeddings']
all_metadatas = all['metadatas']
print(all.keys())

for x_embedding, x_metadata in tqdm(zip(all_embeddings, all_metadatas)):
    x_peek_info_dict = eval(x_metadata['peek_info_dict'])
    x_end_info_dict = eval(x_metadata['end_info_dict'])
    x_pct_dict = eval(x_metadata['pct_dict'])
    x_start_date = x_metadata['start_date']
    x_end_date = x_metadata['end_date']
    x_peek_day = x_metadata['peek_day']
    prevention_zone = get_prevention_zone(x_start_date, '2024-05-01')
    try:
        y = collection.query(query_embeddings=[x_embedding],
                             n_results=top_k_1,
                             where=
                             {
                                 'end_date': {
                                     '$nin': prevention_zone
                                 }
                             })
        y_metadatas = y['metadatas'][0]
        top_k_2_y_indexes = find_top_k_2(x_metadata, y_metadatas, top_k_2)
        print(top_k_2_y_indexes)
    except:
        pass
        # print('something went wrong')

assert len(list1) == len(list2)

import matplotlib.pyplot as plt

# 创建一个包含5个子图的figure
# fig, axes = plt.subplots(3, 2, figsize=(12, 12))
#
# # 散点图(xan,yan)
# axes[0, 0].scatter([x[0] for x in list1], [y[0] for y in list2])
# axes[0, 0].set_title("Scatter Plot (xa, ya)")
# axes[0, 0].set_xlabel("xa")
# axes[0, 0].set_ylabel("ya")
#
# # 散点图(xbn,ybn)
# axes[0, 1].scatter([x[1] for x in list1], [y[1] for y in list2])
# axes[0, 1].set_title("Scatter Plot (xb, yb)")
# axes[0, 1].set_xlabel("xb")
# axes[0, 1].set_ylabel("yb")
#
# # 散点图(xcn,ycn)
# axes[1, 0].scatter([x[2] for x in list1], [y[2] for y in list2])
# axes[1, 0].set_title("Scatter Plot (xc, yc)")
# axes[1, 0].set_xlabel("xc")
# axes[1, 0].set_ylabel("yc")
#
# # 散点图(xdn,ydn)
# axes[1, 1].scatter([x[3] for x in list1], [y[3] for y in list2])
# axes[1, 1].set_title("Scatter Plot (xd, yd)")
# axes[1, 1].set_xlabel("xd")
# axes[1, 1].set_ylabel("yd")
#
# # 散点图(xen,yen)
# axes[2, 0].scatter([x[4] for x in list1], [y[4] for y in list2])
# axes[2, 0].set_title("Scatter Plot (xe, ye)")
# axes[2, 0].set_xlabel("xe")
# axes[2, 0].set_ylabel("ye")
#
# # 调整子图间距并显示图形
# plt.subplots_adjust(hspace=0.5)
# plt.show()

"""
nive = {'peek_info_dict': "{'open': 22686.21, 'close': 22696.01, 'high': 22866.56, 'low': 22627.51, 'vol': 1209932.0}",
        'end_info_dict': "{'open': 22858.79, 'close': 22760.24, 'high': 22858.79, 'low': 22657.95, 'vol': 1043678.9}",
        'pct_dict': "{'open': 0.007607264501210284, 'close': 0.002830012852479498, 'high': -0.0003397975034286065, 'low': 0.0013452651219688922, 'vol': -0.1374069782434054}",
        'start_date': '2014-04-01',
        'end_date': '2014-04-16',
        'peek_day': '2014-04-17'}
"""
