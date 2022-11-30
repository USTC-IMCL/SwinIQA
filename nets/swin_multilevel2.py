# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from option.config import Config
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn.functional as F
from scipy import interpolate

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
import sys
sys.path.append('..')
from nets.crossAtten import Crossatten


# cache each stage results
@lru_cache()
def compute_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0

    for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B,  H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B,  H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim,  num_heads, window_size=(7,7), shift_size=(0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size= self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # print('shift_size',self.shift_size)

        # if self.shift_size > 0:
        #     # calculate attention mask for SW-MSA
        #     H, W = self.input_resolution
        #     print('input_reso',self.input_resolution)
        #     img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        #     h_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     w_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     cnt = 0
        #     for h in h_slices:
        #         for w in w_slices:
        #             img_mask[:, h, w, :] = cnt
        #             cnt += 1
        #
        #     print('img_mask',img_mask.shape)
        #
        #
        #     mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # else:
        #     attn_mask = None
        #
        # self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, mask_matrix):
        # H, W = self.input_resolution
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        B,H,W,C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)
        # x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t =  0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
        # x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[0], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if  any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        # x = x.view(B, H * W, C)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))



        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        B, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # H, W = self.input_resolution
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
   

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim,  depth, num_heads, window_size=(7,7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)


        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0,0) if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # print('ljz',x.shape)
        # calculate attention mask for SW-MSA
        B, C, H, W = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c h w -> b h w c')
        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        attn_mask = compute_mask(Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b h w c -> b c h w')

        return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=320*6, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        # self.patches_resolution = patches_resolution
        # self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        # B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x)#.flatten(2).transpose(1, 2)  # B Ph*Pw C
        # if self.norm is not None:
        #     x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=192, patch_size=1, in_chans=320*6, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=(7,7), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, num_patches = 21*21,**kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # self.patch_embed = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=1)  # b * (320*6) * 21 * 21 --> b * 96 * (21*21)
        num_patches = num_patches
        patches_resolution = [21,21]
        self.patches_resolution = [21,21]

        # # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               # input_resolution=(patches_resolution[0] // (2 ** i_layer),
                               #                   patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        # self.loadswin('../cps/swin_tiny_patch4_window7_224.pth')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def loadswin(self,resume='../cps/swin_tiny_patch4_window7_224.pth'):
        checkpoint=torch.load(resume, map_location='cpu')
        model_dict=self.state_dict()
        new_model_dict={k:v for k,v in checkpoint["model"].items() if k!="head.weight" and k!="head.bias" if k in model_dict}
        model_dict.update(new_model_dict)
        msg = self.load_state_dict(model_dict, strict=False)
        print('feature:',msg)

    def forward(self, x):
        x = self.patch_embed(x)  # batch * (96) * 21*21
        # x = x.reshape(x.shape[0],x.shape[1],-1) # batch * (96) * (21*21)
        # x = x.transpose(0,2,1)  # batch *  (21*21) * (96)
        # print('x',x.shape)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        # print('x',x.shape)
        # print(x.contiguous().shape)
        feat_list = []
        for layer in self.layers:
            x = layer(x.contiguous())
           # print(x.shape)
            feat_list.append(x)

        _, _, ideal_h,ideal_w = feat_list[0].shape
        feat_list = [nn.Upsample(size=(ideal_h,ideal_w), mode='bilinear', align_corners=False)(feat) for feat in feat_list]
        feat_out = torch.cat(feat_list,dim=1)  # ([10, 1920, 28, 28])
        # print(feat_out.shape)

        # print(feat_out.shape)
        # ([10, 192, 28, 28])
        # torch.Size([10, 384, 14, 14])
        # torch.Size([10, 768, 7, 7])
        # torch.Size([10, 768, 7, 7])

        # x = rearrange(x, 'n c h w -> n h w c')
        # x = self.norm(x)
        # x = rearrange(x, 'n h w c -> n c h w')
        # # x = self.norm(x)  # B L C
        # x = self.avgpool(x)  # B C 1
        # x = torch.flatten(x, 1)
        return feat_out

class swin_FR_NR_modified(nn.Module):
    def __init__(self,mode ='2'):
        super().__init__()
        self.swin_extractor = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1, drop_path_rate=0.2)
        self.mode = mode
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.mode == '1': # direct diff regress
            self.head = nn.Sequential(
                        nn.Linear(2112,10),
                        nn.ReLU(),
                        nn.Linear(10,1),
                        nn.Sigmoid()
                    )
        elif self.mode == '2' or self.mode == '3' or self.mode=='4': # cross attention head
            self.crossatten = Crossatten(image_size=(28,28),channels=2112)
            self.head = nn.Sequential(
                nn.Linear(512, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )

        self.NR_head  = nn.Sequential(
                nn.Linear(2112, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )
        self.apply(self._init_weights)
        #self.load_swin()

    def load_swin(self):
        checkpoint = torch.load("../cps/swin_tiny_patch4_window7_224.pth", map_location='cpu')
        model_dict = self.swin_extractor.state_dict()
        new_model_dict = {k: v for k, v in checkpoint["model"].items() if
                          k != "head.weight" and k != "head.bias" and k in model_dict}
        # for k, v in new_model_dict.items():
        #     print(k)
        model_dict.update(new_model_dict)
        msg = self.swin_extractor.load_state_dict(model_dict, strict=False)
        print(msg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self,x):
        return self.swin_extractor(x)

    def forward(self,ref,dist,flag='FR'):
        if flag == 'FR':
            # x:ref, y:dist
            x, y = ref, dist
            x = self.forward_features(x)
            y = self.forward_features(y)  # [10, 1920, 28, 28]
            if self.mode == '1':
                diff_xy = self.avgpool(y - x).squeeze(3).squeeze(2)  # 10 * 1920
                # print('diff',diff_xy.shape)
                x = self.head(diff_xy)
                return x
            elif self.mode == '2':
                cross_atten_feat = self.crossatten(y, x)  # (dist,ref)
                diff_xy = self.avgpool(cross_atten_feat).squeeze(3).squeeze(2)
                # print('cross',diff_xy.shape)
                x = self.head(diff_xy)
                return x
            elif self.mode == '3':
                cross_atten_feat = self.crossatten(y - x, x)
                diff_xy = self.avgpool(cross_atten_feat).squeeze(3).squeeze(2)
                # print('cross',diff_xy.shape)
                x = self.head(diff_xy)
                return x
            elif self.mode == '4':
                cross_atten_feat = self.crossatten((y - x)**2, x)
                diff_xy = self.avgpool(cross_atten_feat).squeeze(3).squeeze(2)
                # print('cross',diff_xy.shape)
                x = self.head(diff_xy)
                return x


        elif flag == 'NR':
            # ref = None / any

            dist = self.forward_features(dist)
            dist = self.avgpool(dist).squeeze(3).squeeze(2)
            # print(dist.shape)
            out = self.NR_head(dist)
            # print(out.shape)
            return out


class swin_FR_NR_modified_clic(nn.Module):
    def __init__(self,mode ='2'):
        super().__init__()
        self.swin_extractor = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1, drop_path_rate=0.2)
        self.mode = mode
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # self.avgpool = F.interpolate()
        if self.mode == '1': # direct diff regress
            self.head = nn.Sequential(
                        nn.Linear(2112,10),
                        nn.ReLU(),
                        nn.Linear(10,1),
                        nn.Sigmoid()
                    )
        elif self.mode == '2' or self.mode == '3' or self.mode=='4': # cross attention head
            self.crossatten = Crossatten(image_size=(28,28),channels=2112)
            self.head = nn.Sequential(
                nn.Linear(512, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )

        # comparer
        self.comparer = nn.Sequential(
            nn.Conv1d(5, 32, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 1, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self.NR_head  = nn.Sequential(
                nn.Linear(2112, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )
        self.apply(self._init_weights)
        self.load_swin()

    def load_swin(self):
        checkpoint = torch.load("../cps/swin_tiny_patch4_window7_224.pth", map_location='cpu')
        model_dict = self.swin_extractor.state_dict()
        new_model_dict = {k: v for k, v in checkpoint["model"].items() if
                          k != "head.weight" and k != "head.bias" and k in model_dict}
        # for k, v in new_model_dict.items():
        #     print(k)
        model_dict.update(new_model_dict)
        msg = self.swin_extractor.load_state_dict(model_dict, strict=False)
        print(msg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self,x):
        return self.swin_extractor(x)

    def forward_once(self,ref,dist,flag='FR'):
        if flag == 'FR':
            # x:ref, y:dist
            x, y = ref, dist
            print("=====test", x.size(), y.size())
            x = self.forward_features(x)
            y = self.forward_features(y)  # [10, 1920, 28, 28]

            print("========x, y", x.size(), y.size())
            if self.mode == '1':
                diff_xy = self.avgpool(y - x).squeeze(3).squeeze(2)  # 10 * 1920
                # print('diff',diff_xy.shape)
                x = self.head(diff_xy)
                return x
            elif self.mode == '2':
                cross_atten_feat = self.crossatten(y, x)  # (dist,ref)
                diff_xy = self.avgpool(cross_atten_feat).squeeze(3).squeeze(2)
                # print('cross',diff_xy.shape)
                x = self.head(diff_xy)
                return x
            elif self.mode == '3':
                cross_atten_feat = self.crossatten(y - x, x)
                diff_xy = self.avgpool(cross_atten_feat).squeeze(3).squeeze(2)
                # print('cross',diff_xy.shape)
                x = self.head(diff_xy)
                return x
            elif self.mode == '4':
                cross_atten_feat = self.crossatten((y - x) ** 2, x)
                diff_xy = self.avgpool(cross_atten_feat).squeeze(3).squeeze(2)
                # diff_xy = F.interpolate(cross_atten_feat, size=(1, 1)).squeeze(3).squeeze(2)
                # print('cross',diff_xy.shape)
                x = self.head(diff_xy)
                return x

        elif flag == 'NR':
            # ref = None / any

            dist = self.forward_features(dist)
            dist = self.avgpool(dist).squeeze(3).squeeze(2)
            # print(dist.shape)
            out = self.NR_head(dist)
            # print(out.shape)
            return out

    def forward(self, dist1,dist2,ref):
        # print((dist1-dist2).min(), (dist1-ref).min())
        # x:ref, y:dist
        score1 = self.forward_once(ref, dist1,'FR')
        score2 = self.forward_once(ref,dist2,'FR')
        # print('score1:', score1, 'score2:', score2)
        d0 = score1.unsqueeze(2)
        d1 = score2.unsqueeze(2)
        input = torch.cat((d0, d1, d0 - d1, d0 / (d1 + 1e-5), d1 / (d0 + 1e-5)), dim=1)
        compare_result = self.comparer(input).squeeze()
        return score1, score2, compare_result


if __name__ == '__main__':
    model = swin_FR_NR_modified(mode='4').cuda()
    model_weight = model.state_dict()
    pretrained_weights = torch.load('../weights/test_weights.pt')['model']
    new_dict = {}
    if 'module' in list(pretrained_weights.keys())[0]:
        for k in pretrained_weights.keys():
            #     print(k)
            if k.replace('module.', '') in model_weight:
                new_dict[k.replace('module.', '')] = pretrained_weights[k]
        # print(new_dict.keys())
        model_weight.update(new_dict)
        model.load_state_dict(model_weight)
    else:
        model.load_state_dict(pretrained_weights)
    with torch.no_grad():
        model.eval()
        input = torch.randn(1,3,224,224).cuda()
        out = model(input,input)
    print(out)
