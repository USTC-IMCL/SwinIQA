import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys

sys.path.append('..')

from nets.transmodules import PreNorm, FeedForward, CAttention, Attention


class Transformer_cross(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(
            nn.ModuleList(
                [
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, CAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

                ]
            )
        )

    def forward(self, x, y):
        for attn, catten, ff in self.layers:
            x = attn(x) + x
            x = catten(x, y) + x
            x = ff(x) + x
        return x

class Crossatten(nn.Module):
    def __init__(self, image_size, channels, dim=512, heads=8, dim_head=64, mlp_dim=128, dropout=0., emb_dropout=0.):
        super().__init__()
        self.h_num, self.w_num = image_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(channels, dim)
        )
        num_patches = image_size[0] * image_size[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_cross(dim, heads, dim_head, mlp_dim, dropout)

    def forward(self, x, y):
        x = self.to_patch_embedding(x)
        y = self.to_patch_embedding(y)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :(n)]
        y += self.pos_embedding[:, :(n)]
        x = self.dropout(x)
        y = self.dropout(y)
        z = self.transformer(x, y)
        z = rearrange(z, 'b (h w) d -> b d h w', h=self.h_num, w=self.w_num)
        return z

class Transformer_cross_v2(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.layers.append(
            nn.ModuleList(
                [
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, CAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

                ]
            )
        )

    def forward(self, x, y):
        for attn, catten, ff in self.layers:
            x = attn(x) + x
            x = catten(x, y) + x
            x = ff(x) + x
        return  nn.LayerNorm(x)

class Crossatten_v2(nn.Module):
    def __init__(self, image_size, channels, dim=512, heads=8, dim_head=64, mlp_dim=128, dropout=0., emb_dropout=0.):
        super().__init__()
        self.h_num, self.w_num = image_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(channels, dim)
        )
        num_patches = image_size[0] * image_size[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_cross_v2(dim, heads, dim_head, mlp_dim, dropout)

    def forward(self, x, y):
        x = self.to_patch_embedding(x)
        y = self.to_patch_embedding(y)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :(n)]
        y += self.pos_embedding[:, :(n)]
        x = self.dropout(x)
        y = self.dropout(y)
        z = self.transformer(x, y)
        z = rearrange(z, 'b (h w) d -> b d h w', h=self.h_num, w=self.w_num)
        return z



if __name__ == "__main__":
    dis = torch.ones((4, 1152, 14, 14)).cuda()
    ref = torch.ones((4, 1152, 14, 14)).cuda()
    model = Crossatten(image_size=(14, 14), channels=1152).cuda()
    out = model(dis, ref)
    print(out.shape)
