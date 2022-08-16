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

        x = self.blocks[0](query, key_value)
        for i in range (1, self.depth):
            x = self.blocks[i](x, x)
        # x = key_value
        # for blk in self.blocks:
        #     x = blk(query, x)

        x = x.permute(0, 3, 1, 2)
        return x


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


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

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
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x


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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse=nn.Conv2d(in_channels+in_channels//2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
            x=self.fuse(x)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        self.img_dim=img_dim
        self.patch_dim=patch_dim
        # ========================================modal1===========================================================
        self.modal1_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.modal1_norm1 = nn.BatchNorm2d(out_channels)
        self.modal1_relu = nn.ReLU(inplace=True)

        self.modal1_encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2, num_layer=2)
        self.modal1_encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2, num_layer=2)
        self.modal1_encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2, num_layer=2)
        self.modal1_encoder4 = EncoderBottleneck(out_channels * 8, out_channels * 16, stride=2, num_layer=2)

        self.vit_img_dim = img_dim[0]*img_dim[1] // patch_dim //patch_dim
        self.modal1_vit = ViT(self.vit_img_dim, out_channels * 16, out_channels * 16,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.modal1_conv2 = nn.Conv2d(out_channels * 16, out_channels * 16, kernel_size=3, stride=1, padding=1)
        self.modal1_norm2 = nn.BatchNorm2d(out_channels * 16)

        # ========================================modal2===========================================================
        self.modal2_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.modal2_norm1 = nn.BatchNorm2d(out_channels)
        self.modal2_relu = nn.ReLU(inplace=True)

        self.modal2_encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2, num_layer=2)
        self.modal2_encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2, num_layer=2)
        self.modal2_encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2, num_layer=2)
        self.modal2_encoder4 = EncoderBottleneck(out_channels * 8, out_channels * 16, stride=2, num_layer=2)

        self.vit_img_dim = img_dim[0] * img_dim[1] // patch_dim // patch_dim
        self.modal2_vit = ViT(self.vit_img_dim, out_channels * 16, out_channels * 16,
                              head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.modal2_conv2 = nn.Conv2d(out_channels * 16, out_channels * 16, kernel_size=3, stride=1, padding=1)
        self.modal2_norm2 = nn.BatchNorm2d(out_channels * 16)
        # ==============================fusion layer====================================================
        num_heads = [2, 2, 4, 8]
        depth = [2, 2, 4, 2]   # 2248
        self.fusion_layer1=NATBlock(dim=out_channels * 2, depth=depth[0], num_heads=num_heads[0], kernel_size=14,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.fusion_layer2 = NATBlock(dim=out_channels * 4, depth=depth[1], num_heads=num_heads[1], kernel_size=14,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.fusion_layer3 = NATBlock(dim=out_channels * 8, depth=depth[2], num_heads=num_heads[2], kernel_size=14,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.fusion_layer4 = NATBlock(dim=out_channels * 16, depth=depth[3], num_heads=num_heads[3], kernel_size=14,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)

        self.modal2_fusion_layer1 = NATBlock(dim=out_channels * 2, depth=depth[0], num_heads=num_heads[0], kernel_size=14,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.modal2_fusion_layer2 = NATBlock(dim=out_channels * 4, depth=depth[1], num_heads=num_heads[1], kernel_size=14,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.modal2_fusion_layer3 = NATBlock(dim=out_channels * 8, depth=depth[2], num_heads=num_heads[2], kernel_size=14,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)
        self.modal2_fusion_layer4 = NATBlock(dim=out_channels * 16, depth=depth[3], num_heads=num_heads[3], kernel_size=14,
                                      mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                                      drop_path=0.1, norm_layer=nn.LayerNorm, layer_scale=None)

    def forward(self, t1_t1ce_now, flair_t2_now):
        # layer1
        modal1_x1 = self.modal1_conv1(t1_t1ce_now)
        modal1_x1 = self.modal1_norm1(modal1_x1)
        modal1_x1 = self.modal1_relu(modal1_x1)
        modal2_x1 = self.modal2_conv1(flair_t2_now)
        modal2_x1 = self.modal2_norm1(modal2_x1)
        modal2_x1 = self.modal2_relu(modal2_x1)

        # layer2
        modal1_x2 = self.modal1_encoder1(modal1_x1)
        modal2_x2 = self.modal2_encoder1(modal2_x1)
        fusion1 = self.fusion_layer1(modal1_x2, modal2_x2)
        modal1_x2 = modal1_x2 + fusion1
        modal2_fusion1 = self.modal2_fusion_layer1(modal2_x2, modal1_x2)
        modal2_x2 = modal2_x2 + modal2_fusion1

        # layer3
        modal1_x3 = self.modal1_encoder2(modal1_x2)
        modal2_x3 = self.modal2_encoder2(modal2_x2)
        fusion2 = self.fusion_layer2(modal1_x3, modal2_x3)
        modal1_x3 = modal1_x3 + fusion2
        modal2_fusion2 = self.modal2_fusion_layer2(modal2_x3, modal1_x3)
        modal2_x3 = modal2_x3 + modal2_fusion2

        # layer4
        modal1_x4 = self.modal1_encoder3(modal1_x3)
        modal2_x4 = self.modal2_encoder3(modal2_x3)
        fusion3 = self.fusion_layer3(modal1_x4, modal2_x4)
        modal1_x4 = modal1_x4 + fusion3
        modal2_fusion3 = self.modal2_fusion_layer3(modal2_x4, modal1_x4)
        modal2_x4 = modal2_x4 + modal2_fusion3

        # layer5
        modal1_x5 = self.modal1_encoder4(modal1_x4)
        modal2_x5 = self.modal2_encoder4(modal2_x4)

        modal1_x5 = self.modal1_vit(modal1_x5)
        modal1_x5 = rearrange(modal1_x5, "b (x y) c -> b c x y", x=self.img_dim[0]//self.patch_dim,
                              y= self.img_dim[1]//self.patch_dim)
        modal1_x5 = self.modal1_conv2(modal1_x5)
        modal1_x5 = self.modal1_norm2(modal1_x5)
        modal1_x5 = self.modal1_relu(modal1_x5)

        modal2_x5 = self.modal2_vit(modal2_x5)
        modal2_x5 = rearrange(modal2_x5, "b (x y) c -> b c x y", x=self.img_dim[0] // self.patch_dim,
                              y=self.img_dim[1] // self.patch_dim)
        modal2_x5 = self.modal2_conv2(modal2_x5)
        modal2_x5 = self.modal2_norm2(modal2_x5)
        modal2_x5 = self.modal2_relu(modal2_x5)

        fusion4 = self.fusion_layer4(modal1_x5, modal2_x5)
        modal1_x5 = modal1_x5 + fusion4
        modal2_fusion4 = self.modal2_fusion_layer4(modal2_x5, modal1_x5)
        modal2_x5 = modal2_x5 + modal2_fusion4

        return modal1_x5, modal1_x1, modal1_x2, modal1_x3, modal1_x4, modal2_x5, modal2_x1, modal2_x2, modal2_x3, modal2_x4


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 16, out_channels * 8)
        self.decoder2 = DecoderBottleneck(out_channels * 8, out_channels*4)
        self.decoder3 = DecoderBottleneck(out_channels * 4, int(out_channels*2))
        self.decoder4 = DecoderBottleneck(int(out_channels*2), int(out_channels))

        self.conv1 = nn.Conv2d(out_channels, class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3,x4):
        x = self.decoder1(x, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        x = self.conv1(x)

        return x


class backbone(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder1 = Decoder(out_channels, class_num)
        
        self.decoder2 = Decoder(out_channels, class_num)

    def forward(self, t1_t1ce_now, flair_t2_now):
        modal1_x5, modal1_x1, modal1_x2, modal1_x3, modal1_x4, \
        modal2_x5, modal2_x1, modal2_x2, modal2_x3, modal2_x4 = self.encoder(t1_t1ce_now, flair_t2_now)
        modal1_output = self.decoder1(modal1_x5, modal1_x1, modal1_x2, modal1_x3, modal1_x4)
        modal2_output = self.decoder2(modal2_x5, modal2_x1, modal2_x2, modal2_x3, modal2_x4)
        return modal1_output, modal2_output


class TranSiam(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.transiam=backbone(img_dim=[224,224],
                          in_channels=2,
                          out_channels=16,
                          head_num=8,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=num_classes)

    def forward(self, t1_t1ce_pre, t1_t1ce_now, t1_t1ce_post, flair_t2_pre, flair_t2_now, flair_t2_post):
        # x = torch.cat((t1_t1ce_now, flair_t2_now), dim=1)
        return self.transiam(t1_t1ce_now, flair_t2_now)




if __name__ == '__main__':
    from torch.autograd import Variable

    x1 = Variable(torch.randn(1, 2, 224, 224))
    x2 = Variable(torch.randn(1, 2, 224, 224))
    net = TranSiam(4)
    _, _ = net(x1, x1, x1, x2,x2,x2)
    from thop import profile

    flops, params = profile(net, inputs=(x1, x1, x1, x2,x2,x2))
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(params)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))