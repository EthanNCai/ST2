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


def get_continuous_days(news_dict_: dict, stock_dict:dict,day_started, n_days):
    start_date = datetime.strptime(day_started, "%Y-%m-%d")
    dates_list = []

    for i in range(n_days + 1):
        next_date = start_date + timedelta(days=i + 1)
        dates_list.append(next_date.strftime("%Y-%m-%d"))
    dates_set = set(dates_list)
    is_valid = dates_set.issubset(set(news_dict_.keys())) and dates_set.issubset(set(stock_dict.keys()))
    return is_valid, dates_list[:-1], dates_list[-2], dates_list[-1]


def generate_weight(n_weights, decay_index=1.5, return_tensor=False, device='cpu'):
    decay_rate = np.log(1 / n_weights) / (n_weights - 1)
    decay_rate *= decay_index

    x_values = np.arange(n_weights)
    y_values = np.exp(-decay_rate * x_values)

    total_sum = np.sum(y_values)
    y_values /= total_sum
    if return_tensor:
        return torch.tensor(y_values).to(torch.float32).to(device)
    else:
        return y_values


def get_change_percentage(end, peek):
    return (peek - end) / end


def get_end_peek_info(stock_dict, end_day, peek_day):
    return stock_dict[end_day], stock_dict[peek_day]


def get_end_peek_pct(stock_dict: dict, end_day, peek_day):
    end_dict: dict = stock_dict[end_day]
    peek_dict: dict = stock_dict[peek_day]
    stock_keys = list(end_dict.keys())
    pct_dict = {}
    for key in stock_keys:
        pct_dict[key] = get_change_percentage(end_dict[key], peek_dict[key])
    return pct_dict


def load_pickle_dict(news_dict_pickle_path):
    import pickle
    with open(news_dict_pickle_path, "rb") as f:
        news_dict_: dict = pickle.load(f)
        print('news_dict_len >>> ', len(news_dict_.keys()))
    return news_dict_


# ----------------------------------------


def get_vec(start_date, stock_dict: dict, news_dict: dict, n_continuous_days: int, teu_, device_):
    news_dict = copy.deepcopy(news_dict)
    is_valid, date_list, end_day, peek_day = get_continuous_days(news_dict
                                                                      , stock_dict
                                                                      , start_date
                                                                      , n_continuous_days)
    if not is_valid:
        return None, None
    news_list_ordered_by_date = [news_dict[date] for date in date_list]
    day_wise_embeddings = torch.concat(list(map(teu_, news_list_ordered_by_date)), dim=0)
    tensor_weights = generate_weight(n_continuous_days, return_tensor=True, device=device_)
    weighted_pooled_embeddings = weighted_pooling(weights_list=tensor_weights, target_tensor=day_wise_embeddings)
    peek_info_dict, end_info_dict = get_end_peek_info(stock_dict, end_day, peek_day)
    pct_dict = get_end_peek_pct(stock_dict, end_day, peek_day)

    return weighted_pooled_embeddings, {"peek_info_dict": str(peek_info_dict),
                                        "end_info_dict": str(end_info_dict),
                                        "pct_dict": str(pct_dict),
                                        "start_date": start_date,
                                        "end_date": end_day,
                                        "peek_day": peek_day},


def test():
    n_days = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    news_dict = load_pickle_dict('../../datas/news_dict.pickle')
    vol_dict = load_pickle_dict('../../stock_fetching/vol_dict.pickle')

    record = []
    import time
    start_time = time.time()
    for date in tqdm(news_dict.keys()):
        embedding, metadata = get_vec(date, vol_dict, news_dict, n_days, device)
        if embedding is None:
            continue
        print((embedding, metadata))

    run_time = time.time() - start_time
    print(record)
    print(f"The loop ran in {run_time} seconds.")


if __name__ == '__main__':
    # test()
    ...
