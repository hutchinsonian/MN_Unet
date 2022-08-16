import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class res_conv_bn(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(res_conv_bn, self).__init__()
        self.res_conv_bn = nn.Sequential(nn.BatchNorm2d(channel_in),
                                         nn.ReLU(),
                                         nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(channel_out),
                                         nn.ReLU(),
                                         nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1))
        self.equal = nn.Sequential(nn.BatchNorm2d(channel_in),
                                   nn.ReLU(),
                                   nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1))

    def forward(self, x):
        out = self.res_conv_bn(x)
        if x.shape[1] != out.shape[1]:
            x = self.equal(x)
        return out + x


# ====================================transform block===============================================
class SelfAttention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('bn', nn.BatchNorm2d(dim_in)),
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x_key_value, x_query, h, w):
        if self.with_cls_token:
            cls_token1, x_key_value = torch.split(x_key_value, [1, h*w], 1)
            cls_token2, x_query = torch.split(x_query, [1, h * w], 1)

        x_key_value = rearrange(x_key_value, 'b (h w) c -> b c h w', h=h, w=w)
        x_query = rearrange(x_query, 'b (h w) c -> b c h w', h=h, w=w)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x_query)
        else:
            q = rearrange(x_query, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x_key_value)
        else:
            k = rearrange(x_key_value, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x_key_value)
        else:
            v = rearrange(x_key_value, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token2, q), dim=1)
            k = torch.cat((cls_token1, k), dim=1)
            v = torch.cat((cls_token1, v), dim=1)

        return q, k, v

    def forward(self, x_key_value, x_query, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x_key_value, x_query, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
            hasattr(module, 'conv_proj_q')
            and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
            hasattr(module, 'conv_proj_k')
            and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
            hasattr(module, 'conv_proj_v')
            and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = x + self.absolute_pos_embed
        if self.norm is not None:
            x = self.norm(x)
        return x


class fuse_transform_Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, img_size, patch_size, in_chans, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop,n_layers=1, stride_kv=1, stride_q=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = nn.ModuleList()
        self.mlp1 = nn.ModuleList()
        self.attn2 = nn.ModuleList()
        self.mlp2 = nn.ModuleList()
        for i in range(n_layers):
            self.attn1.append(SelfAttention(n_embd,n_embd, n_head, attn_pdrop, resid_pdrop, stride_kv=stride_kv, stride_q=stride_q))
            # self.mlp1.append(nn.Sequential(
            #     nn.Linear(n_embd, block_exp * n_embd),
            #     nn.ReLU(True),  # changed from GELU
            #     nn.Linear(block_exp * n_embd, n_embd),
            #     nn.Dropout(resid_pdrop),
            # ))

            self.attn2.append(SelfAttention(n_embd,n_embd, n_head, attn_pdrop, resid_pdrop, stride_kv=stride_kv, stride_q=stride_q))
            # self.mlp2.append(nn.Sequential(
            #     nn.Linear(n_embd, block_exp * n_embd),
            #     nn.ReLU(True), # changed from GELU
            #     nn.Linear(block_exp * n_embd, n_embd),
            #     nn.Dropout(resid_pdrop),
            # ))

        self.patch_embed1 = PatchEmbed(img_size, patch_size, in_chans, n_embd)
        self.patch_embed2 = PatchEmbed(img_size, patch_size, in_chans, n_embd)

        self.fusion1_conv = nn.Sequential(nn.BatchNorm2d(n_embd), nn.ReLU(),
                                          nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1, stride=1))

        self.fusion2_conv = nn.Sequential(nn.BatchNorm2d(n_embd), nn.ReLU(),
                                          nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1, stride=1))

        self.patch_size = patch_size
        self.n_layers = n_layers

    def forward(self, x1, x2):
        input1 = x1
        input2 = x2
        raw_B, raw_C, raw_h, raw_w = input1.size()
        x1 = self.patch_embed1(x1)
        x2 = self.patch_embed1(x2)
        B, T, C = x1.size()

        fusion_x2 = x2
        fusion_x1 = x1
        fusion_x2 = fusion_x2 + self.attn2[0](x1, fusion_x2, int(raw_h / self.patch_size[0]),int(raw_w / self.patch_size[1]))
        fusion_x1 = fusion_x1 + self.attn1[0](x2, fusion_x1, int(raw_h / self.patch_size[0]),int(raw_w / self.patch_size[1]))
        for i in range(1,self.n_layers,1):
            # x2融合到x1
            fusion_x2 = fusion_x2 + self.attn2[i](fusion_x2, fusion_x2, int(raw_h / self.patch_size[0]),int(raw_w / self.patch_size[1]))
            # fusion_x2 = fusion_x2 + self.mlp2[i](self.ln1(fusion_x2))

            # x1融合到x2
            fusion_x1 = fusion_x1 + self.attn1[i](fusion_x1, fusion_x1, int(raw_h/self.patch_size[0]),int(raw_w/self.patch_size[1]))
            # fusion_x1 = fusion_x1 + self.mlp1[i](self.ln2(fusion_x1))

        if T==raw_h*raw_w:
            fusion_x2 = fusion_x2.transpose(1,2).view(B,C,raw_h, raw_w)
            fusion_x1 = fusion_x1.transpose(1, 2).view(B, C, raw_h, raw_w)
        else:
            fusion_x2 = fusion_x2.transpose(1, 2).view(B, C, int(raw_h/self.patch_size[0]), int(raw_w/self.patch_size[1]))
            fusion_x2 = F.interpolate(fusion_x2, scale_factor=(self.patch_size[0], self.patch_size[1]), mode='bilinear',align_corners=True)

            fusion_x1 = fusion_x1.transpose(1, 2).view(B, C, int(raw_h/self.patch_size[0]), int(raw_w/self.patch_size[1]))
            fusion_x1 = F.interpolate(fusion_x1, scale_factor=(self.patch_size[0], self.patch_size[1]), mode='bilinear',align_corners=True)

        return input1 + self.fusion2_conv(fusion_x2), input2 + self.fusion1_conv(fusion_x1)


class channel_fuse_transform_Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, img_size, patch_size, in_chans, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop,n_layers=1, stride_kv=1, stride_q=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = nn.ModuleList()
        self.mlp1 = nn.ModuleList()
        self.attn2 = nn.ModuleList()
        self.mlp2 = nn.ModuleList()
        for i in range(n_layers):
            self.attn1.append(SelfAttention(n_embd,n_embd, n_head, attn_pdrop, resid_pdrop, stride_kv=stride_kv, stride_q=stride_q))

            self.attn2.append(SelfAttention(n_embd,n_embd, n_head, attn_pdrop, resid_pdrop, stride_kv=stride_kv, stride_q=stride_q))

        self.patch_embed1 = PatchEmbed(img_size, patch_size, in_chans, n_embd)
        self.patch_embed2 = PatchEmbed(img_size, patch_size, in_chans, n_embd)

        self.patch_size = patch_size
        self.n_layers = n_layers

    def forward(self, x1, x2):
        input1 = x1
        input2 = x2
        raw_B, raw_C, raw_h, raw_w = input1.size()
        x1 = self.patch_embed1(x1)
        x2 = self.patch_embed1(x2)
        B, T, C = x1.size()

        fusion_x2 = x2 + self.attn2[0](x1, x2, int(raw_h / self.patch_size[0]),int(raw_w / self.patch_size[1]))
        for i in range(1,self.n_layers,1):
            # x2融合到x1
            fusion_x2 = fusion_x2 + self.attn2[i](fusion_x2, fusion_x2, int(raw_h / self.patch_size[0]),int(raw_w / self.patch_size[1]))

        if T==raw_h*raw_w:
            fusion_x2 = fusion_x2.transpose(1,2).view(B,C,raw_h, raw_w)
        else:
            fusion_x2 = fusion_x2.transpose(1, 2).view(B, C, int(raw_h/self.patch_size[0]), int(raw_w/self.patch_size[1]))
            fusion_x2 = F.interpolate(fusion_x2, scale_factor=(self.patch_size[0], self.patch_size[1]), mode='bilinear',align_corners=True)

        return input1 + fusion_x2

class conv_transform_Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, img_size, patch_size, in_chans, n_embd, n_head, attn_pdrop, resid_pdrop,
                 n_layers=1, stride_kv=1, stride_q=1):
        super().__init__()
        self.ln = nn.LayerNorm(in_chans)

        self.attn = nn.ModuleList()
        self.dw_conv = nn.ModuleList()
        self.attn.append(SelfAttention(in_chans, n_embd, n_head, attn_pdrop, resid_pdrop, stride_kv=stride_kv, stride_q=stride_q))
        for i in range(n_layers-1):
            self.attn.append(SelfAttention(in_chans, n_embd, n_head, attn_pdrop, resid_pdrop, stride_kv=stride_kv, stride_q=stride_q))
        for i in range(n_layers):
            self.dw_conv.append(nn.Sequential(
                nn.BatchNorm2d(n_embd), nn.ReLU(),
                nn.Conv2d(n_embd, n_embd, kernel_size=3, padding=1, stride=1)))
                # nn.BatchNorm2d(n_embd), nn.ReLU(),
                # nn.Conv2d(n_embd, n_embd, kernel_size=1, padding=0, stride=1)))

        self.patch_size = patch_size
        self.n_layers = n_layers

        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # self.num_patches = patches_resolution[0] * patches_resolution[1]
        # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, n_embd))
        # trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        raw_B, raw_C, raw_h, raw_w = x.size()

        for i in range(self.n_layers):
            x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
            # x = x + self.absolute_pos_embed
            x = self.ln(x)
            B, T, C = x.size()
            x = x + self.attn[i](x, x, raw_h, raw_w)
            x = x.transpose(1, 2).view(B, C, raw_h, raw_w)
            x = x + self.dw_conv[i](x)

        return x


class conv_cross_attention_Unet(nn.Module):
    def __init__(self, num_classes):
        super(conv_cross_attention_Unet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        # flair and t2
        self.first = nn.Sequential(nn.Conv2d(4, filters[0], 3, 1, 1))

        self.conv1 = res_conv_bn(filters[0], filters[0])
        self.conv2 = res_conv_bn(filters[1], filters[1])
        self.conv3 = res_conv_bn(filters[2], filters[2])
        self.conv4 = nn.Sequential(conv_transform_Block(img_size=[28,28], patch_size=[1,1], in_chans=filters[3],
                                                n_embd=filters[3], n_head=4, attn_pdrop=0.1,
                                                resid_pdrop=0.1, n_layers=2, stride_kv=1, stride_q=1),
                                   nn.BatchNorm2d(filters[3]),
                                   nn.ReLU(),
                                   nn.Conv2d(filters[3], filters[4], kernel_size=3, stride=1, padding=1))

        self.conv5 = conv_transform_Block(img_size=[28, 28], patch_size=[1, 1], in_chans=filters[4],
                                          n_embd=filters[4], n_head=8, attn_pdrop=0.1,
                                          resid_pdrop=0.1, n_layers=2, stride_kv=1, stride_q=1)

        # self.conv4 = res_conv_bn(filters[3], filters[3])
        # self.conv5 = res_conv_bn(filters[4], filters[4])

        self.pool1 = self.downconv(filters[0], filters[1])
        self.pool2 = self.downconv(filters[1], filters[2])
        self.pool3 = self.downconv(filters[2], filters[3])
        # self.pool4 = self.downconv(filters[3], filters[4])

        self.upconv1 = self.upconv(filters[1], filters[0])
        self.upconv2 = self.upconv(filters[2], filters[1])
        self.upconv3 = self.upconv(filters[4], filters[2])
        # self.upconv4 = self.upconv(filters[4], filters[3])

        # self.conv8 = res_conv_bn(filters[4], filters[3])
        self.conv9 = res_conv_bn(filters[3], filters[2])
        self.conv10 = res_conv_bn(filters[2], filters[1])
        self.conv11 = res_conv_bn(filters[1], filters[0])

        self.conv12 = nn.Conv2d(filters[0], num_classes, kernel_size=3, stride=1, padding=1)
        # self.softmax = nn.Softmax(dim=1)

        # t1 and t1ce
        self.modal2_first = nn.Sequential(nn.Conv2d(2, filters[0], 3, 1, 1))

        self.modal2_conv1 = res_conv_bn(filters[0], filters[0])
        self.modal2_conv2 = res_conv_bn(filters[1], filters[1])
        self.modal2_conv3 = res_conv_bn(filters[2], filters[2])
        self.modal2_conv4 = nn.Sequential(conv_transform_Block(img_size=[28, 28], patch_size=[1, 1], in_chans=filters[3],
                                          n_embd=filters[3], n_head=4, attn_pdrop=0.1,
                                          resid_pdrop=0.1, n_layers=2, stride_kv=1, stride_q=1),
                                        nn.BatchNorm2d(filters[3]),
                                        nn.ReLU(),
                                        nn.Conv2d(filters[3], filters[4], kernel_size=3, stride=1, padding=1))

        self.modal2_conv5 = conv_transform_Block(img_size=[28, 28], patch_size=[1, 1], in_chans=filters[4],
                                          n_embd=filters[4], n_head=8, attn_pdrop=0.1,
                                          resid_pdrop=0.1, n_layers=2, stride_kv=1, stride_q=1)
        # self.modal2_conv4 = res_conv_bn(filters[3], filters[3])
        # self.modal2_conv5 = res_conv_bn(filters[4], filters[4])

        self.modal2_pool1 = self.downconv(filters[0], filters[1])
        self.modal2_pool2 = self.downconv(filters[1], filters[2])
        self.modal2_pool3 = self.downconv(filters[2], filters[3])
        # self.modal2_pool4 = self.downconv(filters[3], filters[4])

        self.modal2_upconv1 = self.upconv(filters[1], filters[0])
        self.modal2_upconv2 = self.upconv(filters[2], filters[1])
        self.modal2_upconv3 = self.upconv(filters[4], filters[2])
        # self.modal2_upconv4 = self.upconv(filters[4], filters[3])


        # self.modal2_conv8 = res_conv_bn(filters[4], filters[3])
        self.modal2_conv9 = res_conv_bn(filters[2], filters[2])
        self.modal2_conv10 = res_conv_bn(filters[1], filters[1])
        self.modal2_conv11 = res_conv_bn(filters[0], filters[0])

        self.modal2_conv12 = nn.Conv2d(filters[0], num_classes, kernel_size=3, stride=1, padding=1)

        # self.transform_block1 = fuse_transform_Block(img_size=[88,72], patch_size=[1,1], in_chans=filters[1],
        #                                         n_embd=filters[1], n_head=2, block_exp=4, attn_pdrop=0.1,
        #                                         resid_pdrop=0.1,n_layers=1, stride_kv=4, stride_q=1)
        # self.transform_block2 = fuse_transform_Block(img_size=[44, 36], patch_size=[1, 1], in_chans=filters[2],
        #                                         n_embd=filters[2], n_head=2, block_exp=4, attn_pdrop=0.1,
        #                                         resid_pdrop=0.1,n_layers=1, stride_kv=2, stride_q=1)
        # self.transform_block3 = fuse_transform_Block(img_size=[22, 18], patch_size=[1, 1], in_chans=filters[3],
        #                                         n_embd=filters[3], n_head=4, block_exp=4, attn_pdrop=0.1,
        #                                         resid_pdrop=0.1,n_layers=2)
        self.transform_block4 = fuse_transform_Block(img_size=[28, 28], patch_size=[1, 1], in_chans=filters[4],
                                                n_embd=filters[4], n_head=4, block_exp=4, attn_pdrop=0.1,
                                                resid_pdrop=0.1,n_layers=2)

    def upconv(self, channel_in, channel_out):
        return nn.Sequential(nn.BatchNorm2d(channel_in),
                             nn.ReLU(),
                             nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2))

    def downconv(self, channel_in, channel_out):
        return nn.Sequential(nn.BatchNorm2d(channel_in),
                             nn.ReLU(),
                             nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1))

    def forward(self, t1_t1ce_pre, t1_t1ce_now, t1_t1ce_post, flair_t2_pre, flair_t2_now, flair_t2_post):

        input1 = t1_t1ce_now
        input2 = flair_t2_now
        input1=torch.cat([input1, input2], 1)
        modal1_x = self.first(input1)
        modal1_x1 = self.conv1(modal1_x)
        modal1_x2 = self.pool1(modal1_x1)
        # modal2_x = self.modal2_first(input2)
        # modal2_x1 = self.modal2_conv1(modal2_x)
        # modal2_x2 = self.modal2_pool1(modal2_x1)
        # modal1_x2,modal2_x2 = self.transform_block1(modal1_x2,modal2_x2)

        modal1_x2 = self.conv2(modal1_x2)
        modal1_x3 = self.pool2(modal1_x2)
        # modal2_x2 = self.modal2_conv2(modal2_x2)
        # modal2_x3 = self.modal2_pool2(modal2_x2)
        # modal1_x3, modal2_x3 = self.transform_block2(modal1_x3, modal2_x3)

        modal1_x3 = self.conv3(modal1_x3)
        modal1_x4 = self.pool3(modal1_x3)
        # modal2_x3 = self.modal2_conv3(modal2_x3)
        # modal2_x4 = self.modal2_pool3(modal2_x3)
        # modal1_x4, modal2_x4 = self.transform_block3(modal1_x4, modal2_x4)

        modal1_x5 = self.conv4(modal1_x4)
        # modal2_x5 = self.modal2_conv4(modal2_x4)

        modal1_x5 = self.conv5(modal1_x5)
        # modal2_x5 = self.modal2_conv5(modal2_x5)

        # modal1_x5, modal2_x5 = self.transform_block4(modal1_x5, modal2_x5)

        # t1 and t1ce
        modal1_u3 = self.upconv3(modal1_x5)
        modal1_u3 = torch.cat([modal1_u3, modal1_x3], 1)
        modal1_u3 = self.conv9(modal1_u3)
        modal1_u2 = self.upconv2(modal1_u3)
        modal1_u2 = torch.cat([modal1_u2, modal1_x2], 1)
        modal1_u2 = self.conv10(modal1_u2)
        modal1_u1 = self.upconv1(modal1_u2)
        modal1_u1 = torch.cat([modal1_u1, modal1_x1], 1)
        modal1_u1 = self.conv11(modal1_u1)

        modal1_output = self.conv12(modal1_u1)

        # # flair and t2
        # modal2_u3 = self.modal2_upconv3(modal2_x5)
        # # modal2_u3 = torch.cat([modal2_u3, modal2_x3], 1)
        # modal2_u3 = self.modal2_conv9(modal2_u3)
        # modal2_u2 = self.modal2_upconv2(modal2_u3)
        # # modal2_u2 = torch.cat([modal2_u2, modal2_x2], 1)
        # modal2_u2 = self.modal2_conv10(modal2_u2)
        # modal2_u1 = self.modal2_upconv1(modal2_u2)
        # # modal2_u1 = torch.cat([modal2_u1, modal2_x1], 1)
        # modal2_u1 = self.modal2_conv11(modal2_u1)
        #
        # modal2_output = self.modal2_conv12(modal2_u1)

        return modal1_output, modal1_output


if __name__ == '__main__':
    from torch.autograd import Variable

    x1 = Variable(torch.randn(1, 2, 224, 224))
    x2 = Variable(torch.randn(1, 2, 224, 224))
    net = conv_cross_attention_Unet(4)
    _, _ = net(x1, x1, x1, x2,x2,x2)
    from thop import profile

    flops, params = profile(net, inputs=(x1, x1, x1, x2,x2,x2))
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(params)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))