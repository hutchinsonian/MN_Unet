import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from einops import rearrange, repeat

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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2, num_layer=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2, num_layer=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2, num_layer=2)
        self.encoder4 = EncoderBottleneck(out_channels * 8, out_channels * 16, stride=2, num_layer=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x = self.encoder4(x4)

        return x, x1, x2, x3, x4


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

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3,x4 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3,x4)

        return x, x


class resnet_decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.transunet=backbone(img_dim=[224,224],
                          in_channels=2*2,
                          out_channels=28,
                          head_num=8,
                          mlp_dim=512,
                          block_num=12,
                          patch_dim=16,
                          class_num=num_classes)

    def forward(self, t1_t1ce_pre, t1_t1ce_now, t1_t1ce_post, flair_t2_pre, flair_t2_now, flair_t2_post):
        x = torch.cat((t1_t1ce_now, flair_t2_now), dim=1)
        return self.transunet(x)




if __name__ == '__main__':
    from torch.autograd import Variable

    x1 = Variable(torch.randn(1, 2, 224, 224))
    x2 = Variable(torch.randn(1, 2, 224, 224))
    net = resnet_decoder(4)
    _, _ = net(x1, x1, x1, x2,x2,x2)
    from thop import profile

    flops, params = profile(net, inputs=(x1, x1, x1, x2,x2,x2))
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(params)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))