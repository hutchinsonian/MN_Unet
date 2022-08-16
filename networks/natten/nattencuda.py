"""
Neighborhood Attention PyTorch Module (CUDA only)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import nn
from torch.nn.functional import pad
from timm.models.layers import trunc_normal_
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import os

try:
    this_dir = os.path.dirname(os.path.realpath(__file__))
    from torch.utils.cpp_extension import load
    nattenav_cuda = load(
        'nattenav_cuda', [f'{this_dir}/src/nattenav_cuda.cpp', f'{this_dir}/src/nattenav_cuda_kernel.cu'], verbose=False)
    nattenqkrpb_cuda = load(
        'nattenqkrpb_cuda', [f'{this_dir}/src/nattenqkrpb_cuda.cpp', f'{this_dir}/src/nattenqkrpb_cuda_kernel.cu'], verbose=False)
except:
    try:
        import nattenav_cuda
        import nattenqkrpb_cuda
    except:
        raise RuntimeError("Could not load NATTEN CUDA extension. " +
                           "Please make sure your device has CUDA, the CUDA toolkit for PyTorch is installed, and that you've compiled NATTEN correctly.")


class NATTENAVFunction(Function):
    """
    AV autograd function
    Computes neighborhood attention outputs given attention weights, and values.
    This calls the `AV` kernel.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value):
        attn = attn.contiguous()
        value = value.contiguous()
        out = nattenav_cuda.forward(
                attn, 
                value)
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenav_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_attn, d_value = outputs
        return d_attn, d_value


class NATTENQKRPBFunction(Function):
    """
    QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = nattenqkrpb_cuda.forward(
                query,
                key,
                rpb)
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenqkrpb_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb


class NeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention Module
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # dim: d_model
        # self.head_dim: d_k = d_model/num_heads
        self.head_dim = dim // self.num_heads
        # self.scale: sqrt(d_k)
        self.scale = qk_scale or self.head_dim ** -0.5
        # assert kernel_size > 1 and kernel_size % 2 == 1, \
        #     f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        # assert kernel_size in [3, 5, 7, 9, 11], \
        #     f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, and 11; got {kernel_size}."
        self.kernel_size = kernel_size
        # 该层用于生成qkv矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # rpb: relative position bias 相当于计算atten时softmax里的B
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        # 初始化rpb std=0.02
        trunc_normal_(self.rpb, std=.02)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # 此x为ConvTokenizer的输出，所以channel在最后一个维度
        B, H, W, C = x.shape
        N = H * W
        # NA时窗口为正方形 所以总token数量上等于kernel_size平方
        num_tokens = int(self.kernel_size ** 2)
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        # 此时相当于vit
        if N <= num_tokens:
            if self.kernel_size > W:
                # 在右侧补0
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                # 在下侧补0
                pad_b = self.kernel_size - H
            # x: [batchsize, H, W, embed_dim]
            # 0,0 表示在第一个维度的前和后填充0
            # pad_l,pad_r 表示在第二个维度的前和后填充pad_l和pad_r
            # pad_t,pad_b 表示在第三个维度的前和后填充pad_t和pad_b
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        # x: [batchsize, H, W, embed_dim]
        # self.qkv(x): [batchsize, H, W, embed_dim] -> [batchsize, H, W, 3*embed_dim]
        # embed_dim = d_model = d_k*num_heads = num_heads*head_dim
        # reshape(): [batchsize, H, W, 3*embed_dim] -> [batchsize, H, W, 3, num_heads, head_dim]
        # permute: [batchsize, H, W, 3, num_heads, head_dim] -> [3, batchsize, num_heads, H, W, head_dim]
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        # q k v: [batchsize, num_heads, H, W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)
        # permute: [batchsize, num_heads, H, W, head_dim] -> [batchsize, H, W, num_heads, head_dim]
        # reshape: [batchsize, H, W, num_heads, head_dim] -> [batchsize, H, W, d_model=num_heads*head_dim]
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        # 如果padding了，那么就返回原本的图像
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        return self.proj_drop(self.proj(x))

