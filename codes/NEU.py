from torch import nn
import torch
from transformers import BertTokenizer, BertModel, AdamW
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# # (B, C, L) -> (B, L, C)
# embeddings = embeddings.permute(0, 2, 1)
#
# print(embeddings.shape)
# embeddings = m(embeddings)
#
# # (B, C, L) -> (B, L)
# embeddings = embeddings.view(embeddings.shape[0],embeddings.shape[1])
# print(embeddings.shape)

class NewsExtractionUnit(nn.Module):

    # this is a non-batched unit
    def __init__(self, pre_trained_bert_path, dim_input, dim_output, device='cuda'):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.bert = SentenceTransformer(model_name_or_path=pre_trained_bert_path, device=device)
        self.identity_mlp = torch.nn.Linear(dim_input, dim_output)
        for param in self.bert.parameters():
            param.requires_grad_(False)

    def forward(self, text):
        with torch.no_grad():
            out = self.bert.encode(text, convert_to_tensor=True)
        out = self.identity_mlp(out)
        # ( C, L) -> ( L, C)
        out = out.permute(1, 0)
        out = self.pooling(out)
        # (C, L) -> (L)
        out = out.view(1,out.shape[0],)
        return out


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neu = NewsExtractionUnit('/home/cjz/models/bert-base-chinese/',dim_input=768,dim_output=1024).to(device)
    news = [['你好，这是一个例子','我好吗，这一点都不好'],['我好吗，这一点都不好','我好吗，这一点都不好','我好吗，这一点都不好'],['我好吗，这一点都不好']]
    output = torch.concat([neu(new) for new in news],dim=0)
    print(output.shape)

