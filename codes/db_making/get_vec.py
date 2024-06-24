from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import hashlib
import numpy as np
from datetime import datetime, timedelta
from TEU import TextExtractionUnit
import torch
import copy
from tqdm import tqdm


def weighted_pooling(weights_list: torch.tensor, target_tensor: torch.tensor):
    _device = weights_list.device
    assert len(weights_list) == target_tensor.shape[0]
    assert len(target_tensor.shape) == 2
    # weights_list = torch.tensor(weights_list)
    embedding_size = target_tensor.shape[1]
    weights_tensor = torch.tensor([[weight] * embedding_size for weight in weights_list]).to(_device)
    return sum(weights_tensor * target_tensor)


def get_continuous_days(news_dict_: dict, day_started, n_days):
    start_date = datetime.strptime(day_started, "%Y-%m-%d")
    dates_list = []
    for i in range(n_days + 1):
        next_date = start_date + timedelta(days=i + 1)
        dates_list.append(next_date.strftime("%Y-%m-%d"))
    dates_set = set(dates_list)
    return dates_set.issubset(set(news_dict_.keys())), dates_list[:-1], dates_list[-2], dates_list[-1]


def generate_weight(n_weights, decay_index=1.5, return_tensor=False, device='cpu'):
    decay_rate = np.log(1 / n_weights) / (n_weights - 1)
    decay_rate *= decay_index

    x_values = np.arange(n_weights)
    y_values = np.exp(-decay_rate * x_values)

    total_sum = np.sum(y_values)
    y_values /= total_sum
    if return_tensor:
        return torch.tensor(y_values).to(device)
    else:
        return y_values


def get_peek_change_percentage(vol_dict: dict, end_day: str, peek_day: str):
    end_day_data = vol_dict[end_day]
    peek_day_data = vol_dict[peek_day]
    return (peek_day_data - end_day_data) / end_day_data


def load_pickle_dict(news_dict_pickle_path):
    import pickle
    with open(news_dict_pickle_path, "rb") as f:
        news_dict_: dict = pickle.load(f)
        print('news_dict_len >>> ', len(news_dict_.keys()))
    return news_dict_


# ----------------------------------------

def get_vec(start_date, vol_dict: dict, news_dict: dict, n_continuous_days: int, device_):
    news_dict = copy.deepcopy(news_dict)
    is_continuous, date_list, end_day, peek_day = get_continuous_days(news_dict
                                                                      , start_date
                                                                      , n_continuous_days)
    if not is_continuous or peek_day not in vol_dict or end_day not in vol_dict:
        return None, None
    news_list_ordered_by_date = [news_dict[date] for date in date_list]
    day_wise_embeddings = torch.concat(list(map(teu, news_list_ordered_by_date)), dim=0)
    tensor_weights = generate_weight(n_continuous_days, return_tensor=True, device=device_)
    weighted_pooled_embeddings = weighted_pooling(weights_list=tensor_weights, target_tensor=day_wise_embeddings)
    change_percentage = get_peek_change_percentage(vol_dict, end_day, peek_day)

    return weighted_pooled_embeddings, {"change_percentage": change_percentage, "start_date": start_date}


n_days = 15
embedding_model_path = '../../moka-ai/m3e-large'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
teu = TextExtractionUnit(embedding_model_path, dim_input=1024, dim_output=1024, dropout=0.1, pooling_mode='avg').to(
    device)

news_dict = load_pickle_dict('../../datas/news_dict.pickle')
vol_dict = load_pickle_dict('../../stock_fetching/vol_dict.pickle')

record = []
import time
start_time = time.time()
for date in tqdm(news_dict.keys()):
    embedding, metadata = get_vec(date,vol_dict, news_dict, n_days, device)
    if embedding is None:
        continue
    print((embedding, metadata))

run_time = time.time() - start_time
print(record)
print(f"The loop ran in {run_time} seconds.")
