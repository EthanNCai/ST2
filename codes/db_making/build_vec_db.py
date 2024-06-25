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

n_days = 15
stock_name = 'HSI'


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
embedding_model_path = '../../moka-ai/m3e-large'


client = PersistentClient(path='./test')
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embedding_model_path)

# client.reset()
collection = client.get_or_create_collection(name="a_nice_collection",
                                             embedding_function=embedding_function, )

device = 'cuda' if torch.cuda.is_available() else 'cpu'


teu = TextExtractionUnit(embedding_model_path, dim_input=1024, dim_output=1024, dropout=0.1, pooling_mode='avg').to(
    device)

news_dict = load_pickle_dict('../../datas/news_dict.pickle')
vol_dict = load_pickle_dict('../../stock_fetching/vol_dict.pickle')

embeddings = []
meta_datas = []
texts = []
start_time = time.time()

counter = 0
n_debug = 2

for date in tqdm(news_dict.keys()):
    embedding, metadata = get_vec(date, vol_dict, news_dict, n_days, teu, device)
    if embedding is None:
        continue
    else:
        embeddings.append(embedding.detach().tolist())
        meta_datas.append(metadata)
        texts.append(news_dict[date])
        print(embedding.detach().tolist(), metadata)
    counter += 1
    if counter >= n_debug:
        break

print(len(embeddings), len(meta_datas))
collection.upsert(
    documents=texts,
    embeddings=embeddings,
    metadatas=meta_datas,
    ids=[sha256_str(text[0]) for text in texts]
)

quit()
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
