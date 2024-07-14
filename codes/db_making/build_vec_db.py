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

stage1_windows = 5
stock_name = 'HSI-10'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'mps'

def sort_time_list(date_list: list):
    parsed_dates = [datetime.strptime(date, '%Y-%m-%d') for date in date_list]
    sorted_dates = sorted(parsed_dates)
    sorted_date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]
    return sorted_date_strings


def get_exchange_days(stock_dict: dict):
    date_list = list(stock_dict.keys())
    sorted_time_list = sort_time_list(date_list)
    return sorted_time_list

def sha256_str(str_in: str) -> str:
    return hashlib.md5(str_in.encode('utf-8')).hexdigest()


def load_pickle_dict(news_dict_pickle_path):
    import pickle
    with open(news_dict_pickle_path, "rb") as f:
        news_dict_: dict = pickle.load(f)
        # print('news_dict_len >>> ', len(news_dict_.keys()))
    return news_dict_


news_dict = load_pickle_dict('../../datas/news_dict.pickle')
stock_dict = load_pickle_dict(f'../../stock_fetching/{stock_name}-DICT.pickle')
embedding_model_path = '../../moka-ai/m3e-large'

client = PersistentClient(path='./dbs/n_days_5')
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_path,
                                                                              device='cuda')

# client.reset()
collection = client.get_or_create_collection(name="default",
                                             embedding_function=embedding_function, )

teu = TextExtractionUnit(embedding_model_path, dim_input=1024, dim_output=1024, dropout=0.1, pooling_mode='avg',
                         identity_layer=False, device=device)

embeddings = []
metadatas = []
texts = []
start_time = time.time()
exchange_days_ordered = get_exchange_days(stock_dict)

counter = 0
n_debug = 0

for date in tqdm(news_dict.keys()):
    # if counter >= n_debug:
    #     break
    embedding, metadata = get_vec(date, stock_dict, news_dict, stage1_windows, exchange_days_ordered, teu, device)
    if embedding is None:
        continue
    else:
        first_text = news_dict[date][0]
        embedding = embedding.detach().tolist()
        # embeddings.append(embedding.detach().tolist())
        # metadatas.append(metadata)
        # texts.append(first_text)
        # print(embedding.detach().tolist(), metadata)
    counter += 1
    print('----')
    print(embedding)
    print(first_text)
    print(metadata)
    collection.upsert(
        documents=[first_text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[sha256_str(first_text)]
    )
    print('----')

quit()
