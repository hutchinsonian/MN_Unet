"""
Neighborhood Attention Transformer.
https://arxiv.org/abs/2204.07143

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath
# from mmcv.runner import load_checkpoint
# from mmseg.utils import get_root_logger
# from mmseg.models.builder import BACKBONES
from networks.natten import NeighborhoodAttention



class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(  # (input - kernel + 2*padding) / stride + 1
            # nn.Conv2d: [in_channels, H, W] -> [embed_dim/2, H/2, W/2]
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            # nn.Conv2d: [embed_dim/2, H/2, W/2 -> [embed_dim, H/4, W/4]
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # proj: [batchsize, embed_dim, H, W]
        # permute: [batchsize, H, W, embed_dim]
        # print("before convToker:", x.shape)
        # print("after proj:", self.proj(x).shape)
        x = self.proj(x).permute(0, 2, 3, 1)
        # layer_norm 在channel上做归一化 默认对最后一个维度归一化
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # OVerlapping
        # Conv2d: [dim, H, W] -> [2*dim, H/2, W/2]
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        # x.permute: [batchsize, H, W, embed_dim] -> [batchsize, embed_dim, H, W]
        # reduction: [batchsize, embed_dim, H, W] -> [batchsize, 2*embed_dim, H/2, W/2]
        # reduction().permute: [batchsize, 2*embed_dim, H/2, W/2] -> [batchsize, H/2, W/2, 2*embed_dim]
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # layer_norm 在channel上做归一化 默认对最后一个维度归一化
        x = self.norm(x)
        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)
        x = self.layer(x)
        return x


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


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        # dim为输入图片的dim
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        # attn_drop是丢弃QK结果矩阵中某一个值的概率，相当于更微观
        # proj_drop是丢弃计算得到V的值的概率，相当于更宏观
        self.attn = NeighborhoodAttention(
            dim, kernel_size=13, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # drop_path随机丢弃分支（网络结构）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_ratio控制hidden_features的个数，也就是Mlp的结构
        # nn.Linear只作用在最后一个维度，默认将前面的维度乘一起
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        # 在每一个channel下，对图片的H和W进行缩放
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            # 对NA输出的图片的H和W进行缩放 可训练的参数
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            # 对Mlp输出的图片的H和W进行缩放 可训练的参数
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            # print('x shape:', x.shape)
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        # 对attn的输出 先缩放后决定是否丢弃
        x = shortcut + self.drop_path(self.gamma1 * x)
        # Mlp 先缩放后决定是否丢弃
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class NATLayer97(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        # dim为输入图片的dim
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        # attn_drop是丢弃QK结果矩阵中某一个值的概率，相当于更微观
        # proj_drop是丢弃计算得到V的值的概率，相当于更宏观
        self.attn9 = NeighborhoodAttention(
            dim // 2, kernel_size=9, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn7 = NeighborhoodAttention(
            dim // 2, kernel_size=7, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # drop_path随机丢弃分支（网络结构）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_ratio控制hidden_features的个数，也就是Mlp的结构
        # nn.Linear只作用在最后一个维度，默认将前面的维度乘一起
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        # 在每一个channel下，对图片的H和W进行缩放
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            # 对NA输出的图片的H和W进行缩放 可训练的参数
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            # 对Mlp输出的图片的H和W进行缩放 可训练的参数
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)

            x7 = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x9 = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x7 = self.attn7(x7)
            x9 = self.attn9(x9)
            x = self.conv3(x7.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
                self.conv4(x9.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)

        x7 = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x9 = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x7 = self.attn7(x7)
        x9 = self.attn9(x9)
        x = self.conv3(x7.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
            self.conv4(x9.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 对attn的输出 先缩放后决定是否丢弃
        x = shortcut + self.drop_path(self.gamma1 * x)
        # Mlp 先缩放后决定是否丢弃
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class NATLayer911(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        # dim为输入图片的dim
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        # attn_drop是丢弃QK结果矩阵中某一个值的概率，相当于更微观
        # proj_drop是丢弃计算得到V的值的概率，相当于更宏观
        self.attn9 = NeighborhoodAttention(
            dim // 2, kernel_size=9, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn11 = NeighborhoodAttention(
            dim // 2, kernel_size=11, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # drop_path随机丢弃分支（网络结构）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_ratio控制hidden_features的个数，也就是Mlp的结构
        # nn.Linear只作用在最后一个维度，默认将前面的维度乘一起
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        # 在每一个channel下，对图片的H和W进行缩放
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            # 对NA输出的图片的H和W进行缩放 可训练的参数
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            # 对Mlp输出的图片的H和W进行缩放 可训练的参数
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            # print('x shape:', x.shape)
            x11 = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x9 = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # print('x11 shape:', x11.shape)
            # print('x9 shape:', x9.shape)
            x11 = self.attn11(x11)
            x9 = self.attn9(x9)
            x = self.conv3(x11.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
                self.conv4(x9.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        shortcut = x
        x = self.norm1(x)

        x11 = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x9 = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x11 = self.attn11(x11)
        x9 = self.attn9(x9)
        x = self.conv3(x11.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
            self.conv4(x9.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 对attn的输出 先缩放后决定是否丢弃
        x = shortcut + self.drop_path(self.gamma1 * x)
        # Mlp 先缩放后决定是否丢弃
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class NATLayer3715(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        # dim为输入图片的dim
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        # attn_drop是丢弃QK结果矩阵中某一个值的概率，相当于更微观
        # proj_drop是丢弃计算得到V的值的概率，相当于更宏观
        self.attn7 = NeighborhoodAttention(
            dim // 2, kernel_size=7, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn3 = NeighborhoodAttention(
            dim // 2, kernel_size=3, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn15 = NeighborhoodAttention(
            dim // 2, kernel_size=15, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # drop_path随机丢弃分支（网络结构）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_ratio控制hidden_features的个数，也就是Mlp的结构
        # nn.Linear只作用在最后一个维度，默认将前面的维度乘一起
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        # 在每一个channel下，对图片的H和W进行缩放
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            # 对NA输出的图片的H和W进行缩放 可训练的参数
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            # 对Mlp输出的图片的H和W进行缩放 可训练的参数
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv22 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv33 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            # print('x shape:', x.shape)
            x15 = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x7 = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x3 = self.conv22(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # print('x11 shape:', x11.shape)
            # print('x9 shape:', x9.shape)
            x15 = self.attn15(x15)
            x7 = self.attn7(x7)
            x3 = self.attn3(x3)
            x = self.conv3(x15.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
                self.conv4(x7.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
                self.conv33(x3.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        shortcut = x
        x = self.norm1(x)

        x11 = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x9 = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x11 = self.attn11(x11)
        x9 = self.attn9(x9)
        x = self.conv3(x11.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
            self.conv4(x9.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 对attn的输出 先缩放后决定是否丢弃
        x = shortcut + self.drop_path(self.gamma1 * x)
        # Mlp 先缩放后决定是否丢弃
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class NATLayer357(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        # dim为输入图片的dim
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        # attn_drop是丢弃QK结果矩阵中某一个值的概率，相当于更微观
        # proj_drop是丢弃计算得到V的值的概率，相当于更宏观
        self.attn7 = NeighborhoodAttention(
            dim // 2, kernel_size=7, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn3 = NeighborhoodAttention(
            dim // 2, kernel_size=3, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn5 = NeighborhoodAttention(
            dim // 2, kernel_size=5, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # drop_path随机丢弃分支（网络结构）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_ratio控制hidden_features的个数，也就是Mlp的结构
        # nn.Linear只作用在最后一个维度，默认将前面的维度乘一起
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        # 在每一个channel下，对图片的H和W进行缩放
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            # 对NA输出的图片的H和W进行缩放 可训练的参数
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            # 对Mlp输出的图片的H和W进行缩放 可训练的参数
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv22 = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim // 2),
                                   nn.ReLU(inplace=True))
        self.conv33 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            # print('x shape:', x.shape)
            x5 = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x7 = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x3 = self.conv22(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # print('x11 shape:', x11.shape)
            # print('x9 shape:', x9.shape)
            x5 = self.attn5(x5)
            x7 = self.attn7(x7)
            x3 = self.attn3(x3)
            x = self.conv3(x5.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
                self.conv4(x7.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
                self.conv33(x3.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        shortcut = x
        x = self.norm1(x)

        x11 = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x9 = self.conv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x11 = self.attn11(x11)
        x9 = self.attn9(x9)
        x = self.conv3(x11.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + \
            self.conv4(x9.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 对attn的输出 先缩放后决定是否丢弃
        x = shortcut + self.drop_path(self.gamma1 * x)
        # Mlp 先缩放后决定是否丢弃
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        # dim为输入图片的dim
        self.dim = dim
        # depth为每个NAT Block 有多少个 NAT layer
        self.depth = depth
        # nn.ModuleList内部没有实现forward函数
        self.blocks = nn.ModuleList([
            # NATLayer(dim=dim,
            #          num_heads=num_heads, kernel_size=kernel_size,
            #          mlp_ratio=mlp_ratio,
            #          qkv_bias=qkv_bias, qk_scale=qk_scale,
            #          drop=drop, attn_drop=attn_drop,
            #          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            #          norm_layer=norm_layer,
            #          layer_scale=layer_scale)
            # NATLayer911(dim=dim,
            #             num_heads=num_heads,
            #             mlp_ratio=mlp_ratio,
            #             qkv_bias=qkv_bias, qk_scale=qk_scale,
            #             drop=drop, attn_drop=attn_drop,
            #             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            #             norm_layer=norm_layer,
            #             layer_scale=layer_scale)
            NATLayer3715(dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


# @BACKBONES.register_module()
class NAT(nn.Module):
    def __init__(self,
                 num_classes,
                 embed_dim=32,                   # 第一个NAT Block输出的channel
                 mlp_ratio=3.0,                  # Mlp隐藏层神经元比例
                 depths=[3, 4, 18, 5],            # 有几个NAT Block 每个Block有几个Layer
                 num_heads=[2, 4, 8, 16],        # atten head的个数
                 drop_path_rate=0.2,             # 丢弃分支的概率
                 in_chans=1,                     # 输入图片的维度
                 kernel_size=11,                 # atten窗口大小
                 out_indices=(0, 1, 2, 3),       # 对每个Block的输出feature数进行索引
                 qkv_bias=True,                  # qkv是否使用rpb
                 qk_scale=None,                  # /sqrt(dk)
                 drop_rate=0,                   # 丢弃计算得到V的概率
                 attn_drop_rate=0,              # 丢弃QK矩阵中某一个值的概率
                 norm_layer=nn.LayerNorm,        # LN
                 frozen_stages=-1,               # 冻结权重
                 pretrained=None,                # 是否使用预训练权重
                 layer_scale=None,               # 在每一个channel下，对NA输出的图片的H和W进行缩放
                 **kwargs):
        super().__init__()
        # 有几个NAT Block
        self.num_levels = len(depths)
        # ConvTokenizer输出的维度
        self.embed_dim = embed_dim
        # [dim, 2*dim, 4*dim] 每一个NAT Block输出的dim
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        # mlp_ratio*in_features为Mlp隐藏层神经元的个数
        self.mlp_ratio = mlp_ratio
        # ConvTokenizer: [in_chans, H, W] -> [embed_dim, H/4, W/4]
        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        # drop_rate: 丢弃计算得到V的概率
        self.pos_drop = nn.Dropout(p=drop_rate)
        # 生成每一个Block的dpr
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(dim=int(embed_dim * 2 ** i),
                             depth=depths[i],
                             num_heads=num_heads[i],
                             kernel_size=kernel_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             norm_layer=norm_layer,
                             downsample=(i < self.num_levels - 1),  # 最后一个NAT Block没有downsampling
                             layer_scale=layer_scale)
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.init_weights(pretrained)

        n1 = embed_dim
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.decoder1 = DecoderBottleneck(filters[3], filters[2])
        self.decoder2 = DecoderBottleneck(filters[2], filters[1])
        self.decoder3 = DecoderBottleneck(filters[1], filters[0])
        self.decoder4 = DecoderBottleneck(filters[0], filters[0], scale_factor=4)

        self.convLast = nn.Conv2d(filters[0], num_classes, kernel_size=3, stride=1, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            # self.patch_embed = ConvTokenizer 不更新权重
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.network[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(NAT, self).train(mode)
        self._freeze_stages()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            pass
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    # ConvTokenizer
    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, level in enumerate(self.levels):
            # NAT Block 输出为 downsample(x_output), x_output
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(xo)
                # x_out: [batchsize, H, W, embed_dim]
                # permute: [batchsize, embed_dim, H, W]
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        # outs 为每一个Block的输出list（没有下采样的）
        return outs

    def forward(self, x):
        # print("forward:", x.shape)
        x = self.forward_embeddings(x)
        # print("after embeddings:", x.shape)
        outs = self.forward_tokens(x)
        # print("after tokens:", outs[3].shape)
        # decoder1: 1/16
        outputs = self.decoder1(outs[3])
        outputs = outputs + outs[2]
        # decoder2: 1/8
        outputs = self.decoder2(outputs)
        outputs = outputs + outs[1]
        # decoder3: 1/4
        outputs = self.decoder3(outputs)
        outputs = outputs + outs[0]
        # decoder2: 1/1
        outputs = self.decoder4(outputs)
        outputs = self.convLast(outputs)
        # print("end:", outputs.shape)
        return outputs

    def forward_features(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)


if __name__ == '__main__':
    from torch.autograd import Variable

    x1 = Variable(torch.randn(1, 1, 512, 512)).cuda()
    net = NAT(2).cuda()
    _ = net(x1)
    from thop import profile

    flops, params = profile(net, inputs=(x1,))
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(params)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
