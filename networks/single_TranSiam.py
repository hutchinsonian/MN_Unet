import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# ====================================transform block===============================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) / self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x

class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class conv(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, mlp_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, mlp_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, embedding_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        # self.mlp = MLP(embedding_dim, mlp_dim)
        self.conv = conv(embedding_dim, mlp_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.batch_norm1 = nn.BatchNorm2d(embedding_dim)
        self.batch_norm2 = nn.BatchNorm2d(embedding_dim)
        self.Relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.layernorm=nn.LayerNorm(embedding_dim)
    def forward(self, x):
        x = self.layernorm(x)
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x

        x = rearrange(x, "b (x y) c -> b c x y", x=28, y=28)
        x = self.batch_norm1(x)

        _x = self.conv(x)
        _x = self.batch_norm2(_x)
        # _x = self.mlp(x)
        x = x + _x
        x = self.Relu(x)
        x = rearrange(x, "b c x y -> b (x y) c", x=28, y=28)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x

class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        # token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
        #                batch_size=batch_size)
        #
        # patches = torch.cat([token, project], dim=1)
        # patches += self.embedding[:tokens + 1, :]

        x = self.dropout(project)
        x = self.transformer(x)
        # x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x

# class conv_transform_Block(nn.Module):
#     """ an unassuming Transformer block """
#
#     def __init__(self, img_size, patch_size, in_chans, n_embd, n_head, attn_pdrop, resid_pdrop,
#                  n_layers=1, stride_kv=1, stride_q=1):
#         super().__init__()
#         self.ln = nn.LayerNorm(in_chans)
#
#         self.attn = nn.ModuleList()
#         self.dw_conv = nn.ModuleList()
#         self.attn.append(MultiHeadAttention(n_embd, n_head))
#             # SelfAttention(in_chans, n_embd, n_head, attn_pdrop, resid_pdrop, stride_kv=stride_kv, stride_q=stride_q))
#         for i in range(n_layers-1):
#             self.attn.append(MultiHeadAttention(n_embd, n_head))
#                 # SelfAttention(in_chans, n_embd, n_head, attn_pdrop, resid_pdrop, stride_kv=stride_kv, stride_q=stride_q))
#         for i in range(n_layers):
#             self.dw_conv.append(nn.Sequential(
#                 nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1, stride=1),
#                 nn.BatchNorm2d(n_embd), nn.ReLU(inplace=True),
#                 ))
#
#         self.patch_size = patch_size
#         self.n_layers = n_layers
#
#         # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         # self.num_patches = patches_resolution[0] * patches_resolution[1]
#         # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, n_embd))
#         # trunc_normal_(self.absolute_pos_embed, std=.02)
#
#     def forward(self, x):
#         raw_B, raw_C, raw_h, raw_w = x.size()
#
#         for i in range(self.n_layers):
#             x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
#             # x = x + self.absolute_pos_embed
#             x = self.ln(x)
#             B, T, C = x.size()
#             x = x + self.attn[i](x)
#             x = x.transpose(1, 2).view(B, C, raw_h, raw_w)
#             x = x + self.dw_conv[i](x)
#         return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride=stride
        self.in_channels=in_channels
        self.out_channels = out_channels

        self.equal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.res_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.res_conv_bn(x)
        if self.stride==2 or self.in_channels!=self.out_channels:
            x=self.equal(x)
        return self.relu(x1+x)


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_layer=1):
        super().__init__()
        self.stride=stride
        self.num_layer=num_layer
        self.layer=nn.ModuleList()

        if stride==2:
            self.layer.append(Bottleneck(in_channels, out_channels, stride=2))
        else:
            self.layer.append(Bottleneck(in_channels, out_channels, stride=1))

        for i in range(num_layer-1):
            self.layer.append(Bottleneck(out_channels, out_channels, stride=1))

    def forward(self, x):
        for layer in self.layer:
            x=layer(x)
        return x

class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
            x=self.fuse(x)

        x = self.layer(x)
        return x


class single_TranSiam(nn.Module):
    def __init__(self, num_classes):
        super(single_TranSiam, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.first = nn.Conv2d(4, filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(filters[0], filters[1], stride=2, num_layer=2)  # 112
        self.encoder2 = EncoderBottleneck(filters[1], filters[2], stride=2, num_layer=2)  # 56
        self.encoder3 = EncoderBottleneck(filters[2], filters[3], stride=2, num_layer=2)  # 28

        self.vit = ViT(28, filters[3], filters[3],
                       head_num=8, mlp_dim=int(filters[3]*0.65), block_num=12, patch_dim=1, classification=False)

        self.decoder1 = DecoderBottleneck(filters[3]+filters[2], filters[2])
        self.decoder2 = DecoderBottleneck(filters[2]+filters[1], filters[1])
        self.decoder3 = DecoderBottleneck(filters[1]+filters[0], filters[0])

        self.conv12 = nn.Conv2d(filters[0], num_classes, kernel_size=3, stride=1, padding=1)

    def upconv(self, in_channels, out_channels):
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1_t1ce_pre, t1_t1ce_now, t1_t1ce_post, flair_t2_pre, flair_t2_now, flair_t2_post):

        input1 = t1_t1ce_now
        input2 = flair_t2_now
        input1=torch.cat([input1, input2], 1)
        x = self.first(input1)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = self.vit(x)

        x = rearrange(x, "b (x y) c -> b c x y", x=28, y=28)

        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.conv12(x)

        return x, x


if __name__ == '__main__':
    from torch.autograd import Variable

    x1 = Variable(torch.randn(1, 2, 224, 224))
    x2 = Variable(torch.randn(1, 2, 224, 224))
    net = single_TranSiam(4)
    _, _ = net(x1, x1, x1, x2,x2,x2)
    from thop import profile

    flops, params = profile(net, inputs=(x1, x1, x1, x2,x2,x2))
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(params)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))