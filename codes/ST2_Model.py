import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from TEU import TextExtractionUnit


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


## class mean_pooling ~

class ST2(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=1, dim_head=64,
                 dropout=0., emb_dropout=0., text_emb_model_path='../google-bert/bert-base-chinese/',
                 text_embeddings_dim=1024):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            # identity layer
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(dim))
        self.cls_token_text = nn.Parameter(torch.randn(text_embeddings_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.teu = TextExtractionUnit(text_emb_model_path, dim_input=768, dim_output=text_embeddings_dim).to('cuda')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.to('cuda')

    def forward(self, time_series, texts):
        # PART 1 -> encode the texts using TE (text extractor)

        # PART 2 -> encode the time series patches
        patch_embeddings = self.to_patch_embedding(time_series)
        # patched and flattened x -> torch.Size([1, 1024])
        n_batch, n_channel, _ = patch_embeddings.shape

        cls_patch_tokens = repeat(self.cls_token_patch, 'd -> b d', b=n_batch)
        cls_texts_tokens = repeat(self.cls_token_text, 'd -> b d', b=n_batch)

        patch_embeddings, ps_patch = pack([cls_patch_tokens, patch_embeddings], 'b * d')
        # cls_patch_tokens    -> torch.Size([1, 1024])
        # x             -> torch.Size([1, 16, 1024])
        # packed_x      -> torch.Size([1, 17, 1024])
        # ps -> pack information that can be used to later de-packing operation

        # PART 3 -> concat texts embeddings and time series embeddings

        text_embeddings = torch.concat([self.teu(text) for text in texts], dim=0)
        text_embeddings, ps_text = pack([cls_texts_tokens, text_embeddings], 'b * d')

        print('text_embeddings.shape', text_embeddings.shape)
        print('patch_embeddings.shape', patch_embeddings.shape)

        # PART 4 -> Positional embedding
        concat_embeddings = torch.add(patch_embeddings, text_embeddings)

        concat_embeddings += self.pos_embedding[:, :(n_channel + 1)]
        # pos_embedding                 -> torch.Size([1, 17, 1024])
        # pos_embedding[:, :(n_channel + 1)]    -> torch.Size([1, 17, 1024])

        # PART 5 -> final output

        concat_embeddings = self.dropout(concat_embeddings)
        concat_embeddings = self.transformer(concat_embeddings)
        # x -> torch.Size([1, 17, 1024])

        cls_patch_tokens, _ = unpack(concat_embeddings, ps_patch, 'b * d')

        return self.mlp_head(cls_patch_tokens)


def test_case():
    batch_size = 32
    time_step = 256
    patch_size = 16
    patch_token_dim = 1024

    st2 = ST2(
        seq_len=time_step,
        patch_size=patch_size,
        num_classes=time_step,
        channels=1,
        dim=patch_token_dim,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
        text_embeddings_dim=1024,
        text_emb_model_path='../google-bert/bert-base-chinese/'
    )
    time_points_batch_2 = torch.concat([torch.randn(time_step).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32),
                                torch.randn(time_step).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)],dim=0).to('cuda')
    print('time_points_batch_2.shape', time_points_batch_2.shape)

    texts_batch_2 = [['你好1', '你好2'], ['你好3', '你好4', '你好5']]
    out = st2(time_points_batch_2, texts_batch_2)

# test_case()
