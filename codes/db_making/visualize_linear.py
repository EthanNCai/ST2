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

client = PersistentClient(path='./test2')
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embedding_model_path)

collection = client.get_or_create_collection(name="a_nice__collection",
                                             embedding_function=embedding_function, )

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# teu = TextExtractionUnit(embedding_model_path, dim_input=1024, dim_output=1024, dropout=0.1, pooling_mode='avg',
#                          identity_layer=False).to(device)

news_dict = load_pickle_dict('../../datas/news_dict.pickle')
vol_dict = load_pickle_dict('../../stock_fetching/vol_dict.pickle')

xs = []
ys = []

for date in tqdm(news_dict.keys()):
    text = news_dict[date][0]
    result = collection.query(query_texts=[' '], where={'start_date': date})
    if len(result['distances'][0]) == 0:
        continue
    else:
        x = result['metadatas'][0][0]['change_percentage']
        print(x)
