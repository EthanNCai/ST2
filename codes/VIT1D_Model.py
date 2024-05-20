import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


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

class VIT1D_Model(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=1, dim_head=64,
                 dropout=0., emb_dropout=0.):
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
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, time_series):
        x = self.to_patch_embedding(time_series)
        # patched and flattened x -> torch.Size([1, 1024])
        n_batch, n_channel, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=n_batch)
        # print(x.shape)
        x, ps = pack([cls_tokens, x], 'b * d')
        # cls_tokens    -> torch.Size([1, 1024])
        # x             -> torch.Size([1, 16, 1024])
        # packed_x      -> torch.Size([1, 17, 1024])
        # ps -> pack information that can be used to later de-packing operation

        x += self.pos_embedding[:, :(n_channel + 1)]
        # pos_embedding                 -> torch.Size([1, 17, 1024])
        # pos_embedding[:, :(n_channel + 1)]    -> torch.Size([1, 17, 1024])

        x = self.dropout(x)
        x = self.transformer(x)
        # x -> torch.Size([1, 17, 1024])

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)


def test_case():
    batch_size = 32
    time_step = 256
    patch_size = 16
    patch_token_dim = 1024

    vit1d = VIT1D_Model(
        seq_len=time_step,
        patch_size=patch_size,
        num_classes=time_step,
        channels=1,
        dim=patch_token_dim,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    time_points = torch.randn(time_step).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    # tensor -> (batch, channels, len)
    semantic_info = []
    out = vit1d(time_points, semantic_info)

# test_case()
