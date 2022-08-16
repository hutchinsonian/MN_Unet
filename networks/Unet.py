import torch
import torch.nn as nn
import numpy as np

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

        n1 = 24
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1*32]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(1, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.Conv6 = conv_block(filters[4], filters[5])


        self.Up6 = up_conv(filters[5], filters[4])
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # x: [bs, 1, 512, 512]
        # conv1(x): [bs, 24, 512, 512]
        e1 = self.Conv1(x)
        # e2: [bs, 24, 256, 256]
        e2 = self.Maxpool1(e1)
        # conv2(e2): [bs, 24x2, 256, 256]
        e2 = self.Conv2(e2)
        # e3: [bs, 24x2, 128, 128]
        e3 = self.Maxpool2(e2)
        # conv3(e3): [bs, 24x4, 128, 128]
        e3 = self.Conv3(e3)
        # e4: [bs, 24x4, 64, 64]
        e4 = self.Maxpool3(e3)
        # conv4(e4): [bs, 24x8, 64, 64]
        e4 = self.Conv4(e4)
        # e5: [bs, 24x8, 32, 32]
        e5 = self.Maxpool4(e4)
        # conv5: [bs, 24x16, 32, 32]
        e5 = self.Conv5(e5)
        # e6: [bs, 24x16, 16, 16]
        e6 = self.Maxpool5(e5)
        # conv6: [bs, 24x32, 16, 16]
        e6 = self.Conv6(e6)

        # e6: [bs, 24x32, 16, 16]
        # d6: [bs, 24x16, 32, 32]
        d6 = self.Up6(e6, e5)
        # d5: [bs, 24x8, 64, 64]
        d5 = self.Up5(d6, e4)
        # d4: [bs, 24x4, 128, 128]
        d4 = self.Up4(d5, e3)
        # d3: [bs, 24x2, 256, 256]
        d3 = self.Up3(d4, e2)
        # d2: [bs, 24x1, 512, 512]
        d2 = self.Up2(d3, e1)
        # out: [bs, 2, 512, 512]
        out1 = self.Conv(d2)

        return out1, out1

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