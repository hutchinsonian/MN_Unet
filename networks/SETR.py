import torch.nn as nn
import torch
from collections import OrderedDict

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return x + position_embeddings


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs

class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, in_planes, out_planes=None, mid_planes=None, stride=1):
        super(PreActBottleneck, self).__init__()
        out_planes = out_planes or in_planes
        mid_planes = mid_planes or out_planes // 4

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv1x1(in_planes, mid_planes)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv2 = conv3x3(mid_planes, mid_planes, stride)
        self.bn3 = nn.BatchNorm2d(mid_planes)
        self.conv3 = conv1x1(mid_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != out_planes:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(in_planes, out_planes, stride)

    def forward(self, x):
        out = self.relu(self.bn1(x))

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        return out + residual


class ResNetV2Model(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843):
        super(ResNetV2Model, self).__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines. Do not format
        # fmt: off
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('pad', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=64*wf, out_planes=256*wf, mid_planes=64*wf))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=256*wf, out_planes=256*wf, mid_planes=64*wf)) for i in range(2, block_units[0] + 1)],
        ))
        self.conv3 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=256*wf, out_planes=512*wf, mid_planes=128*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=512*wf, out_planes=512*wf, mid_planes=128*wf)) for i in range(2, block_units[1] + 1)],
        ))
        self.conv4 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=512*wf, out_planes=1024*wf, mid_planes=256*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=1024*wf, out_planes=1024*wf, mid_planes=256*wf)) for i in range(2, block_units[2] + 1)],
        ))
        self.conv5 = nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(in_planes=1024*wf, out_planes=2048*wf, mid_planes=512*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_planes=2048*wf, out_planes=2048*wf, mid_planes=512*wf)) for i in range(2, block_units[3] + 1)],
        ))

        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.BatchNorm2d(2048*wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
            ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
        ]))
        # fmt: on

    def forward(self, x, include_conv5=False, include_top=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if include_conv5:
            x = self.conv5(x)
        if include_top:
            x = self.head(x)

        if include_top and include_conv5:
            assert x.shape[-2:] == (1, 1,)
            return x[..., 0, 0]

        return x


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

class HybridSegmentationTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_classes,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        include_conv5=False,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        positional_encoding_type="learned",
        backbone='r50x1',
    ):
        super(HybridSegmentationTransformer, self).__init__()

        assert embedding_dim % num_heads == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.include_conv5 = include_conv5
        self.backbone = backbone
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.patch_dim = patch_dim
        self.num_classes = num_classes

        # self.backbone_model, self.flatten_dim = self.configure_backbone()
        # self.projection_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        self.projection_encoding = nn.Conv2d(num_channels, self.embedding_dim, kernel_size=patch_dim, stride=patch_dim)

        self.decoder_dim = int(self.img_dim[0]*self.img_dim[1]/self.patch_dim/self.patch_dim)
        if self.include_conv5:
            self.decoder_dim = int(self.img_dim[0]*self.img_dim[1]/self.patch_dim/self.patch_dim)

        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.decoder_dim, self.embedding_dim, self.decoder_dim
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.out_layer1=nn.Sequential(nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(),
        )

        self.out_layer2 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1)

        # =====================================decoder===============================
        n1 = embedding_dim
        filters = [n1//8, n1//4, n1//2, n1]
        self.upconv0 = nn.Sequential(self.upconv(filters[0], filters[0]),
                                     self.upconv(filters[0], filters[0]))
        self.upconv1 = self.upconv(filters[1], filters[0])
        self.upconv2 = self.upconv(filters[2], filters[1])
        self.upconv3 = self.upconv(filters[3], filters[2])

        self.conv9 = res_conv_bn(filters[2], filters[2])
        self.conv10 = res_conv_bn(filters[1], filters[1])
        self.conv11 = res_conv_bn(filters[0], filters[0])

        self.conv12 = nn.Conv2d(filters[0], num_classes, kernel_size=3, stride=1, padding=1)

    def upconv(self, channel_in, channel_out):
        return nn.Sequential(nn.BatchNorm2d(channel_in),
                             nn.ReLU(),
                             nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2))

    def encode(self, x):
        # apply bit backbone
        # x = self.backbone_model(x, include_conv5=self.include_conv5)
        x = self.projection_encoding(x)
        x=x.view(x.size(0), x.size(1),-1)
        x=x.permute(0,2,1)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        return x, intmd_x

    def decode(self, x):
        x = self._reshape_output(x)

        # t1 and t1ce
        modal1_u3 = self.upconv3(x)
        modal1_u3 = self.conv9(modal1_u3)
        modal1_u2 = self.upconv2(modal1_u3)
        modal1_u2 = self.conv10(modal1_u2)
        modal1_u1 = self.upconv1(modal1_u2)
        modal1_u1 = self.conv11(modal1_u1)
        # modal1_u0 = self.upconv0(modal1_u1)
        modal1_output = self.conv12(modal1_u1)

        # x = self.out_layer1(x)
        #
        # x = nn.Upsample(scale_factor=self.patch_dim, mode='bilinear')(x)
        #
        # x = self.out_layer2(x)

        return modal1_output

    def forward(self, x):
        encoder_output, intmd_encoder_outputs = self.encode(x)
        decoder_output = self.decode(encoder_output)

        return decoder_output,decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim[0] / self.patch_dim),
            int(self.img_dim[1] / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    # def configure_backbone(self):
    #     """
    #     Current support offered for all BiT models
    #     KNOWN_MODELS in https://github.com/google-research/big_transfer/blob/master/bit_pytorch/models.py
    #     expects model name of style 'r{depth}x{width}'
    #     where depth in [50, 101, 152]
    #     where width in [1,2,3,4]
    #     """
    #     backbone = self.backbone
    #
    #     splits = backbone.split('x')
    #     model_name = splits[0]
    #     width_factor = int(splits[1])
    #
    #     if model_name in ['r50', 'r101'] and width_factor in [2, 4]:
    #         return ValueError(
    #             "Invalid Configuration of models -- expect 50x1, 50x3, 101x1, 101x3"
    #         )
    #     elif model_name == 'r152' and width_factor in [1, 3]:
    #         return ValueError(
    #             "Invalid Configuration of models -- expect 152x2, 152x4"
    #         )
    #
    #     block_units_dict = {
    #         'r50': [3, 4, 6, 3],
    #         'r101': [3, 4, 23, 3],
    #         'r152': [3, 8, 36, 3],
    #     }
    #     block_units = block_units_dict.get(model_name, [3, 4, 6, 3])
    #     model = ResNetV2Model(
    #         block_units, width_factor, head_size=self.num_classes
    #     )
    #
    #     if self.num_channels == 3:
    #         flatten_dim = 1024 * width_factor
    #     if self.include_conv5:
    #         flatten_dim *= 2
    #
    #     return model, flatten_dim


class setr(nn.Module):
    def __init__(self,num_classes):
        super(setr, self).__init__()

        self.setr=HybridSegmentationTransformer(img_dim=[224,224],
        patch_dim=8,
        num_classes=num_classes,
        num_channels=2*2,
        embedding_dim=256,
        num_heads=8,
        num_layers=12,
        hidden_dim=256*4,
        include_conv5=False,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        positional_encoding_type="learned",
        backbone='r50x1')

    def forward(self, t1_t1ce_pre, t1_t1ce_now, t1_t1ce_post, flair_t2_pre, flair_t2_now, flair_t2_post):
        x = torch.cat((t1_t1ce_now, flair_t2_now), dim=1)
        return self.setr(x)


if __name__ == '__main__':
    from torch.autograd import Variable

    x1 = Variable(torch.randn(1, 2, 224, 224))
    x2 = Variable(torch.randn(1, 2, 224, 224))
    net = setr(4)
    _, _ = net(x1, x1, x1, x2, x2, x2)
    from thop import profile

    flops, params = profile(net, inputs=(x1, x1, x1, x2, x2, x2))
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(params)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))