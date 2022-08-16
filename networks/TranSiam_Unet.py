import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mmseg.models.builder import BACKBONES
from networks.natten import NeighborhoodAttention


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


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1, drop_path=0.1,
                 norm_layer=nn.LayerNorm, layer_scale=None):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.mlp = MLP(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))

    def forward(self, query, key_value):
        shortcut = key_value
        query = self.norm1(query)
        key_value = self.norm2(key_value)

        x = self.attn(query, key_value)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class NATBlock(nn.Module):    # 输入和输出都是 B H W C
    def __init__(self, dim, depth, num_heads, kernel_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer,
                     layer_scale=layer_scale)
            for i in range(depth)])

    def forward(self, query, key_value):
        query = query.permute(0, 2, 3, 1)
        key_value = key_value.permute(0, 2, 3, 1)
        x = key_value
        for blk in self.blocks:
            x = blk(query, x)
        x = x.permute(0, 3, 1, 2)
        return x


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Conv2d(in_ch + in_ch // 2, in_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_concat=None):
        x = self.upsample(x)
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
            x = self.fuse(x)
        x = self.layer(x)
        return x


class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1*32]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(2, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.Conv6 = conv_block(filters[4], filters[5])

        self.modal2_Conv1 = conv_block(2, filters[0])
        self.modal2_Conv2 = conv_block(filters[0], filters[1])
        self.modal2_Conv3 = conv_block(filters[1], filters[2])
        self.modal2_Conv4 = conv_block(filters[2], filters[3])
        self.modal2_Conv5 = conv_block(filters[3], filters[4])
        self.modal2_Conv6 = conv_block(filters[4], filters[5])

        self.Up6 = up_conv(filters[5], filters[4])
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])

        self.modal2_Up6 = up_conv(filters[5], filters[4])
        self.modal2_Up5 = up_conv(filters[4], filters[3])
        self.modal2_Up4 = up_conv(filters[3], filters[2])
        self.modal2_Up3 = up_conv(filters[2], filters[1])
        self.modal2_Up2 = up_conv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.modal2_Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

        # ==============================fusion layer====================================================
        num_heads = [2, 2, 4, 8, 4]
        depth = [2, 2, 4, 8, 4]
        self.fusion_layer1 = NATBlock(dim=filters[0], depth=depth[0], num_heads=num_heads[0], kernel_size=7,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.fusion_layer2 = NATBlock(dim=filters[1], depth=depth[1], num_heads=num_heads[1], kernel_size=7,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.fusion_layer3 = NATBlock(dim=filters[2], depth=depth[2], num_heads=num_heads[2], kernel_size=7,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.fusion_layer4 = NATBlock(dim=filters[3], depth=depth[3], num_heads=num_heads[3], kernel_size=7,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.fusion_layer5 = NATBlock(dim=filters[4], depth=depth[4], num_heads=num_heads[4], kernel_size=7,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)

    def forward(self, t1_t1ce_pre, t1_t1ce_now, t1_t1ce_post, flair_t2_pre, flair_t2_now, flair_t2_post):
        e1 = self.Conv1(t1_t1ce_now)
        e2 = self.Maxpool1(e1)
        modal2_e1 = self.modal2_Conv1(flair_t2_now)
        modal2_e2 = self.Maxpool1(modal2_e1)
        fusion1 = self.fusion_layer1(e2, modal2_e2)
        e2 = e2 + fusion1

        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        modal2_e2 = self.modal2_Conv2(modal2_e2)
        modal2_e3 = self.Maxpool2(modal2_e2)
        fusion2 = self.fusion_layer2(e3, modal2_e3)
        e3 = e3 + fusion2

        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        modal2_e3 = self.modal2_Conv3(modal2_e3)
        modal2_e4 = self.Maxpool3(modal2_e3)
        fusion3 = self.fusion_layer3(e4, modal2_e4)
        e4 = e4+ fusion3

        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        modal2_e4 = self.modal2_Conv4(modal2_e4)
        modal2_e5 = self.Maxpool4(modal2_e4)
        fusion4 = self.fusion_layer4(e5, modal2_e5)
        e5 = e5 + fusion4

        e5 = self.Conv5(e5)
        e6 = self.Maxpool5(e5)
        modal2_e5 = self.modal2_Conv5(modal2_e5)
        modal2_e6 = self.Maxpool5(modal2_e5)
        fusion5 = self.fusion_layer5(e6, modal2_e6)
        e6 = e6 + fusion5

        e6 = self.Conv6(e6)
        modal2_e6 = self.modal2_Conv6(modal2_e6)

        #===========modal1===========
        d6 = self.Up6(e6, e5)
        d5 = self.Up5(d6, e4)
        d4 = self.Up4(d5, e3)
        d3 = self.Up3(d4, e2)
        d2 = self.Up2(d3, e1)
        out1 = self.Conv(d2)

        # ======================== modal2===========================
        modal2_d6 = self.modal2_Up6(modal2_e6, modal2_e5)
        modal2_d5 = self.modal2_Up5(modal2_d6, modal2_e4)
        modal2_d4 = self.modal2_Up4(modal2_d5, modal2_e3)
        modal2_d3 = self.modal2_Up3(modal2_d4, modal2_e2)
        modal2_d2 = self.modal2_Up2(modal2_d3, modal2_e1)
        out2 = self.modal2_Conv(modal2_d2)

        return out1, out2

if __name__ == '__main__':
    from torch.autograd import Variable

    x1 = Variable(torch.randn(1, 2, 224, 224))
    x2 = Variable(torch.randn(1, 2, 224, 224))
    net = Unet(4)
    _, _ = net(x1, x1, x1, x2,x2,x2)
    from thop import profile

    flops, params = profile(net, inputs=(x1, x1, x1, x2,x2,x2))
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(params)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))