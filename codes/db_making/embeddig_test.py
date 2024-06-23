from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import hashlib
import numpy as np
from datetime import datetime, timedelta
from TEU import TextExtractionUnit
import torch
continuous_day = 15
embedding_model_path = '..\..\moka-ai\m3e-large'


def sha256_str(str_in: str) -> str:
    return hashlib.md5(str_in.encode('utf-8')).hexdigest()


def generate_weight(n_weights, decay_rate):
    decay_rate = np.log(1 / n_weights) / (n_weights - 1)
    decay_rate *= 1.5

    x_values = np.arange(n_weights)
    y_values = np.exp(-decay_rate * x_values)  # y值是指数衰减序列

    total_sum = np.sum(y_values)
    y_values /= total_sum
    return y_values


def daily_sorted(news_dict_pickle_path):
    import pickle
    with open(news_dict_pickle_path, "rb") as f:
        news_dict_: dict = pickle.load(f)
        print('news_dict_len >>> ', len(news_dict_.keys()))
    return news_dict_


def get_continuous_days(news_dict_: dict, day_started, n_days):
    start_date = datetime.strptime(day_started, "%Y-%m-%d")
    dates_list = []
    for i in range(n_days):
        next_date = start_date + timedelta(days=i + 1)
        dates_list.append(next_date.strftime("%Y-%m-%d"))
    dates_set = set(dates_list)
    return dates_set.issubset(set(news_dict_.keys())), dates_list


news_dict = daily_sorted('../../datas/news_dict.pickle')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# teu = TextExtractionUnit('/home/cjz/models/bert-base-chinese/', dim_input=768, dim_output=1024).to(device)
teu = TextExtractionUnit(embedding_model_path, dim_input=1024, dim_output=1024, dropout=0.1,pooling_mode='avg').to(device)

for key in news_dict.keys():
    is_continuous, date_list = get_continuous_days(news_dict, key, continuous_day)
    if not is_continuous:
        continue
    news_list_ordered_by_date = [news_dict[date] for date in date_list]
    day_wise_embeddings = torch.concat([teu(new) for new in news_list_ordered_by_date], dim=0)
    # output = for day_wise_embedding in day_wise_embeddings
    # #
    # print(output.shape)
    quit()


client = PersistentClient(path='./test')
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()

# client.reset()
collection = client.get_or_create_collection(name="a_nice_collection",
                                             embedding_function=embedding_function, )



source_texts = [("挺好的，我感觉这个很棒", "2021-01-02"),
                ("不错的，这个确实很好，我很推荐", "2021-01-03"),
                ("这玩意太垃圾了把", "2021-01-04"),
                ("这个东西真的是十分的糟糕", "2021-01-05"),
                ("很好，我喜欢这个东西", "2021-01-06")]


# 增加数据
texts = [text for text, _ in source_texts]
dates = [{"date": date} for _, date in source_texts]
ids = [sha256_str(text) for text in texts]

collection.add(
    documents=texts,
    ids=ids,
    metadatas=dates
)

# 查数据
result = collection.query(
    query_texts=["不错，我觉得挺好", "好棒啊"],
    n_results=1,
)
print(result)

# 更新
# collection.update(
#     documents=texts,
#     ids=[sha256_str(text) for text in texts]
# )

new_text = ['我觉得不好不坏', '一般般把']
collection.upsert(
    documents=new_text,
    ids=[sha256_str(text) for text in new_text]
)

print(result)

# 删除
del_target_document = ['我觉得不好不坏']
collection.delete(
    ids=[sha256_str(text) for text in del_target_document]
)

print(collection.peek()['documents'])
#
