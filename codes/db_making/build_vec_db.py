from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import hashlib
import numpy as np
from datetime import datetime, timedelta
from TEU import TextExtractionUnit
import torch

n_continuous_day = 15

stock_name = ''


def sha256_str(str_in: str) -> str:
    return hashlib.md5(str_in.encode('utf-8')).hexdigest()


def load_pickle_dict(news_dict_pickle_path):
    import pickle
    with open(news_dict_pickle_path, "rb") as f:
        news_dict_: dict = pickle.load(f)
        # print('news_dict_len >>> ', len(news_dict_.keys()))
    return news_dict_


news_dict = load_pickle_dict('../../datas/news_dict.pickle')
vol_dict = load_pickle_dict('../../stock_fetching/vol_dict.pickle')


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
