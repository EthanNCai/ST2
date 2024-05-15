import numpy as np
import pandas as pd
from torch import nn
import torch
from sentence_transformers import SentenceTransformer


# model = SentenceTransformer('moka-ai/m3e-base', cache_folder='./weights/embedding') # 第一次需要下载， 可以指定下载位置，默认为用户cache目录
model = SentenceTransformer(model_name_or_path='/home/cjz/models/bert-base-chinese/', device="cuda")  # 加载本地模型


# print(model.max_seq_length)

#Our sentences we like to encode


batch_size = 2

news = [
    ['今天股票要大涨',
    '今天股票感觉势头不错',
    'A股今天有潜力'],
    ['A股今天有潜力']
]

m = nn.AdaptiveMaxPool1d(output_size=1)

embeddings = np.array(model.encode(news_) for news_ in news)

# (C, L) -> (B, L, C)
# embeddings = torch.tensor(embeddings).unsqueeze(0)

# (B, C, L) -> (B, L, C)
embeddings = embeddings.permute(0, 2, 1)

print(embeddings.shape)
embeddings = m(embeddings)

# (B, C, L) -> (B, L)
embeddings = embeddings.view(embeddings.shape[0],embeddings.shape[1])
print(embeddings.shape)

