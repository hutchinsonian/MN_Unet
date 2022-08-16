import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import functional as F

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
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
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
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

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

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

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

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
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

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
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

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class flow_field(nn.Module):
    def __init__(self, trans_inplane, cnn_inplane):
        super(flow_field, self).__init__()
        self.trans_flow_make = nn.Sequential(
            nn.Conv2d(trans_inplane * 2, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.PReLU())
        self.cnn_flow_make = nn.Sequential(
            nn.Conv2d(cnn_inplane * 2, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.PReLU())
        self.cnn_down = nn.Sequential(
            nn.Conv2d(cnn_inplane, trans_inplane, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(trans_inplane),
            nn.PReLU())

        self.trans_input_map = nn.Sequential(
            nn.BatchNorm2d(trans_inplane),
            nn.Conv2d(trans_inplane, trans_inplane, kernel_size=1, stride=1),
            nn.BatchNorm2d(trans_inplane),
            nn.PReLU())
        self.trans_up_map = nn.Sequential(
            nn.BatchNorm2d(trans_inplane),
            nn.Conv2d(trans_inplane, cnn_inplane, kernel_size=1, stride=1),
            nn.BatchNorm2d(cnn_inplane),
            nn.PReLU())
        self.cnn_map = nn.Sequential(
            nn.Conv2d(cnn_inplane, cnn_inplane, kernel_size=1, stride=1),
            nn.BatchNorm2d(cnn_inplane),
            nn.PReLU())
        self.norm_layer = nn.LayerNorm(trans_inplane)

    def forward(self, trans_input, cnn_input):
        # ===========================trans_fuse=======================
        B, C, H, W = trans_input.shape
        cnn_down = self.cnn_down(cnn_input)
        trans_input_map = self.trans_input_map(trans_input)
        # trans_flow = self.trans_flow_make(torch.cat([trans_input_map, cnn_down], dim=1))
        # trans_flow_warp = self.flow_warp(cnn_down, trans_flow, cnn_down.size()[2:])
        # trans_fuse = trans_flow_warp+trans_input
        trans_fuse = trans_input_map + cnn_down

        trans_fuse = trans_fuse.view(B, C, H*W)
        trans_fuse = trans_fuse.permute(0, 2, 1)
        trans_fuse = self.norm_layer(trans_fuse)

        # =============================cnn===============================
        trans_up = F.interpolate(trans_input, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        trans_up_map = self.trans_up_map(trans_up)
        cnn_map = self.cnn_map(cnn_input)
        # cnn_flow = self.cnn_flow_make(torch.cat([trans_up_map, cnn_map], dim=1))
        # cnn_flow_warp = self.flow_warp(trans_up_map, cnn_flow, trans_up_map.size()[2:])
        # cnn_fuse = cnn_flow_warp+cnn_input
        cnn_fuse = trans_up_map+cnn_map

        return trans_fuse,cnn_fuse

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=False)
        return output


class SwinTransformerSys(nn.Module):
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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        self.patch_merge = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.patch_merge.append(PatchMerging((patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                                                 dim=int(embed_dim * 2 ** i_layer), norm_layer=norm_layer))
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.patch_expand = nn.ModuleList()
        for i_layer in range(self.num_layers-1):
            self.concat_back_dim.append(nn.Linear(2*int(embed_dim*2**(self.num_layers-2-i_layer)),
            int(embed_dim*2**(self.num_layers-2-i_layer))))

            self.patch_expand.append(PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
            patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer))

            layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-2-i_layer)),
                            input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-2-i_layer)),
                                                patches_resolution[1] // (2 ** (self.num_layers-2-i_layer))),
                            depth=depths[(self.num_layers-2-i_layer)],
                            num_heads=num_heads[(self.num_layers-2-i_layer)],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:(self.num_layers-2-i_layer)]):sum(depths[:(self.num_layers-2-i_layer) + 1])],
                            norm_layer=norm_layer,
                            upsample=None,
                            use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)

        # ==============================================CNN=====================================================
        cnn_channel = embed_dim//6
        filters = [cnn_channel, cnn_channel*2,cnn_channel*4,cnn_channel*8,cnn_channel*16,cnn_channel*32]
        self.cnn_stem = nn.ModuleList()
        self.cnn_layer1 = nn.ModuleList()
        self.cnn_layer2 = nn.ModuleList()
        self.cnn_layer3 = nn.ModuleList()
        self.cnn_layer4 = nn.ModuleList()
        self.cnn_decoder1 = nn.ModuleList()
        self.cnn_decoder2 = nn.ModuleList()
        self.cnn_decoder3 = nn.ModuleList()
        self.cnn_decoder4 = nn.ModuleList()
        self.cnn_output = nn.ModuleList()
        for i in range(4):
            self.maxpool = nn.MaxPool2d(2, stride=2)
            # self.cnn_stem.append(nn.Sequential(nn.Conv2d(in_chans, cnn_channel, 7, 2, 3),nn.BatchNorm2d(cnn_channel), nn.ReLU(),
            #                     nn.Conv2d(cnn_channel, cnn_channel, 3, 2, 1),nn.BatchNorm2d(cnn_channel), nn.ReLU()))
            self.cnn_stem.append(nn.Sequential(nn.Conv2d(in_chans, filters[0], 7, 2, 3),nn.BatchNorm2d(filters[0]), nn.ReLU()))
            self.cnn_layer1.append(nn.Sequential(nn.Conv2d(filters[0], filters[0], 3, 1, 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(),
                                       nn.Conv2d(filters[0], filters[0], 3, 1, 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU()))
            self.cnn_layer2.append(nn.Sequential(nn.Conv2d(filters[0], filters[1], 3, 1, 1),
                                       nn.BatchNorm2d(filters[1]),
                                       nn.ReLU(),
                                       nn.Conv2d(filters[1], filters[1], 3, 1, 1),
                                       nn.BatchNorm2d(filters[1]),
                                       nn.ReLU()))
            self.cnn_layer3.append(nn.Sequential(nn.Conv2d(filters[1], filters[2], 3, 1, 1),
                                        nn.BatchNorm2d(filters[2]),
                                        nn.ReLU(),
                                        nn.Conv2d(filters[2], filters[2], 3, 1, 1),
                                        nn.BatchNorm2d(filters[2]),
                                        nn.ReLU()))
            self.cnn_layer4.append(nn.Sequential(nn.Conv2d(filters[2], filters[3], 3, 1, 1),
                                                 nn.BatchNorm2d(filters[3]),
                                                 nn.ReLU(),
                                                 nn.Conv2d(filters[3], filters[3], 3, 1, 1),
                                                 nn.BatchNorm2d(filters[3]),
                                                 nn.ReLU()))

            self.cnn_decoder1.append(nn.Sequential(nn.Conv2d(filters[3]+filters[2], filters[2], 3, 1, 1),
                                          nn.BatchNorm2d(filters[2]),
                                          nn.ReLU(),
                                          nn.Conv2d(filters[2], filters[2], 3, 1, 1),
                                          nn.BatchNorm2d(filters[2]),
                                          nn.ReLU()))
            self.cnn_decoder2.append(nn.Sequential(nn.Conv2d(filters[2]+filters[1], filters[1], 3, 1, 1),
                                          nn.BatchNorm2d(filters[1]),
                                          nn.ReLU(),
                                          nn.Conv2d(filters[1], filters[1], 3, 1, 1),
                                          nn.BatchNorm2d(filters[1]),
                                          nn.ReLU()))
            self.cnn_decoder3.append(nn.Sequential(nn.Conv2d(filters[1]+filters[0], filters[0], 3, 1, 1),
                                          nn.BatchNorm2d(filters[0]),
                                          nn.ReLU(),
                                          nn.Conv2d(filters[0], filters[0], 3, 1, 1),
                                          nn.BatchNorm2d(filters[0]),
                                          nn.ReLU()))
            self.cnn_decoder4.append(nn.Sequential(nn.Conv2d(filters[0], int(filters[0] / 2), 3, 1, 1),
                                                   nn.BatchNorm2d(int(filters[0] / 2)),
                                                   nn.ReLU(),
                                                   nn.Conv2d(int(filters[0] / 2), int(filters[0] / 2), 3, 1, 1),
                                                   nn.BatchNorm2d(int(filters[0] / 2)),
                                                   nn.ReLU()))
            self.cnn_output.append(nn.Conv2d(in_channels=int(filters[0] / 2),out_channels=self.num_classes,kernel_size=1,bias=False))
        # =================================================fuse=================================================
        self.fuse1 = flow_field(embed_dim,filters[0])
        self.fuse2 = flow_field(embed_dim*2, filters[1])
        self.fuse3 = flow_field(embed_dim*4, filters[2])

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

    def cnn_layer(self,layer,input):
        cnn_layer = []
        for count in range(4):
            cnn_layer.append(layer[count](input[count]))
        return cnn_layer

    def fuse(self,cnn_input, trans_input, k):
        count = 0
        hei_cnn_layer = []
        for i in range(2):
            wid_cnn_layer = []
            for j in range(2):
                wid_cnn_layer.append(cnn_input[count])
                count = count + 1
            hei_cnn_layer.append(torch.cat(wid_cnn_layer,-1))
        cnn_layer = torch.cat(hei_cnn_layer,-2)

        B, L, C = trans_input.shape
        if k==1:
            trans_input = trans_input.view(B, 56, 56, C)
            trans_input = trans_input.permute(0, 3, 1, 2)  # B,C,H,W
            trans_fuse, cnn_fuse = self.fuse1(trans_input, cnn_layer)
        if k==2:
            trans_input = trans_input.view(B, 28, 28, C)
            trans_input = trans_input.permute(0, 3, 1, 2)  # B,C,H,W
            trans_fuse, cnn_fuse = self.fuse2(trans_input, cnn_layer)
        if k==3:
            trans_input = trans_input.view(B, 14, 14, C)
            trans_input = trans_input.permute(0, 3, 1, 2)  # B,C,H,W
            trans_fuse, cnn_fuse = self.fuse3(trans_input, cnn_layer)
        # cnn
        cnn_feature = []
        b,c,h,w = cnn_fuse.shape
        for i in range(2):
            for j in range(2):
                cnn_feature.append(cnn_fuse[:, :, int(i * h/2):int((i + 1) * h/2), int(j * w/2):int((j + 1) * w/2)])

        return trans_fuse,cnn_feature

    def cnn_conca(self, input1,input2):
        final=[]
        for i in range(4):
            final.append(torch.cat([input1[i], input2[i]], 1))
        return final

    def interpolate(self, input,scale=2):
        final = []
        for i in range(4):
            final.append(F.interpolate(input[i], scale_factor=(scale, scale), mode='bilinear',align_corners=True))
        return final

    def cnn_pool(self, input):
        final = []
        for i in range(4):
            final.append(self.maxpool(input[i]))
        return final

    def forward_features(self, input):
        x = self.patch_embed(input)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # cnn
        count=0
        cnn_stem=[]
        for i in range(2):
            for j in range(2):
                cnn_stem.append(self.cnn_stem[count](input[:, :, int(i*112):int((i+1)*112), int(j*112):int((j+1)*112)]))
                count = count+1
        cnn_layer1 = self.cnn_layer(self.cnn_layer1,cnn_stem)
        # trans
        trans_layer1 = self.layers[0](x)
        # fuse1
        trans_fuse1, cnn_fuse1 = self.fuse(cnn_layer1,trans_layer1,1)
        # down
        cnn_down1 = self.cnn_pool(cnn_fuse1)
        trans_merge1 = self.patch_merge[0](trans_fuse1)

        # cnn
        cnn_layer2 = self.cnn_layer(self.cnn_layer2, cnn_down1)
        # trans
        trans_layer2 = self.layers[1](trans_merge1)
        # fuse2
        trans_fuse2, cnn_fuse2 = self.fuse(cnn_layer2, trans_layer2, 2)
        # down
        cnn_down2 = self.cnn_pool(cnn_fuse2)
        trans_merge2 = self.patch_merge[1](trans_fuse2)

        # cnn
        cnn_layer3 = self.cnn_layer(self.cnn_layer3, cnn_down2)
        # trans
        trans_layer3 = self.layers[2](trans_merge2)
        # fuse3
        trans_fuse3, cnn_fuse3 = self.fuse(cnn_layer3, trans_layer3, 3)
        # down
        cnn_down3 = self.cnn_pool(cnn_fuse3)
        trans_merge3 = self.patch_merge[2](trans_fuse3)

        # cnn
        cnn_layer4 = self.cnn_layer(self.cnn_layer4, cnn_down3)
        # trans
        trans_layer4 = self.layers[3](trans_merge3)
        # fuse4
        # trans_fuse4, cnn_fuse4 = self.fuse(cnn_layer4, trans_layer4, 4)

        # ========================decoder===================================================
        # cnn
        cnn_up1 = self.interpolate(cnn_layer4, scale=2)
        cnn_decoder1 = self.cnn_conca(cnn_up1, cnn_fuse3)
        cnn_decoder1 = self.cnn_layer(self.cnn_decoder1, cnn_decoder1)
        # trans
        trans_decoder1 = self.patch_expand[0](trans_layer4)
        trans_decoder1 = torch.cat([trans_decoder1, trans_fuse3], -1)
        trans_decoder1 = self.concat_back_dim[0](trans_decoder1)
        trans_decoder1 = self.layers_up[0](trans_decoder1)

        # cnn
        cnn_up2 = self.interpolate(cnn_decoder1,scale=2)
        cnn_decoder2 = self.cnn_conca(cnn_up2, cnn_fuse2)
        cnn_decoder2 = self.cnn_layer(self.cnn_decoder2, cnn_decoder2)
        # trans
        trans_decoder2 = self.patch_expand[1](trans_decoder1)
        trans_decoder2 = torch.cat([trans_decoder2, trans_fuse2], -1)
        trans_decoder2 = self.concat_back_dim[1](trans_decoder2)
        trans_decoder2 = self.layers_up[1](trans_decoder2)

        # cnn
        cnn_up3 = self.interpolate(cnn_decoder2,scale=2)
        cnn_decoder3 = self.cnn_conca(cnn_up3, cnn_fuse1)
        cnn_decoder3 = self.cnn_layer(self.cnn_decoder3, cnn_decoder3)
        # trans
        trans_decoder3 = self.patch_expand[2](trans_decoder2)
        trans_decoder3 = torch.cat([trans_decoder3, trans_fuse1], -1)
        trans_decoder3 = self.concat_back_dim[2](trans_decoder3)
        trans_decoder3 = self.layers_up[2](trans_decoder3)
  
        return cnn_decoder3, trans_decoder3

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)
            
        return x

    def forward(self, x):
        # x, x_downsample = self.forward_features(x)
        # x = self.forward_up_features(x,x_downsample)
        # x = self.up_x4(x)
        cnn, trans = self.forward_features(x)
        trans = self.up_x4(trans)

        cnn = self.interpolate(cnn,scale=2)
        cnn = self.cnn_layer(self.cnn_decoder4, cnn)
        cnn = self.cnn_layer(self.cnn_output, cnn)

        return trans,cnn

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

    # Dencoder and Skip connection
    # def forward_up_features(self, x, x_downsample):
    #         for inx, layer_up in enumerate(self.layers_up):
    #             if inx == 0:
    #                 x = layer_up(x)
    #             else:
    #                 x = torch.cat([x, x_downsample[3 - inx]], -1)
    #                 x = self.concat_back_dim[inx](x)
    #                 x = layer_up(x)
    #
    #         x = self.norm_up(x)  # B L C
    #
    #         return x


