from torch import nn
import torch
from transformers import BertTokenizer, BertModel, AdamW
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class TextExtractionUnit(nn.Module):

    # this is a non-batched unit
    def __init__(self, pre_trained_bert_path, dim_input, dim_output,dropout,pooling_mode, device='cuda'):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool1d(output_size=1) if pooling_mode == "max" else nn.AdaptiveAvgPool1d(output_size=1)
        self.bert = SentenceTransformer(model_name_or_path=pre_trained_bert_path, device=device)
        self.identity_mlp = torch.nn.Linear(dim_input, dim_output)
        self.layer_norm = nn.LayerNorm(dim_output)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        for param in self.bert.parameters():
            param.requires_grad_(False)

    def forward(self, text):
        with torch.no_grad():
            out = self.bert.encode(text, convert_to_tensor=True)
        out = self.dropout(out)
        out = self.identity_mlp(out)
        # out = self.layer_norm(out)
        out = self.gelu(out)
        # ( C, L) -> ( L, C)
        out = out.permute(1, 0)
        out = self.pooling(out)
        # out = self.gelu(out)
        # (C, L) -> (L)
        out = out.view(1, out.shape[0], )
        return out


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # teu = TextExtractionUnit('/home/cjz/models/bert-base-chinese/', dim_input=768, dim_output=1024).to(device)
    teu = TextExtractionUnit('../google-bert/bert-base-chinese/', dim_input=768, dim_output=1024).to(device)
    news = [['你好，这是一个例子', '我好吗，这一点都不好'],
            ['我好吗，这一点都不好', '我好吗，这一点都不好', '我好吗，这一点都不好'], ['我好吗，这一点都不好']]
    output = torch.concat([teu(new) for new in news], dim=0)
    print(output.shape)

# test()