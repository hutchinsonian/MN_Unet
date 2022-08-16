import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x_key_value, x_query):
        B, T, C = x_key_value.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x_key_value).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x_query).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x_key_value).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


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


class transform_Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, img_size, patch_size, in_chans, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop,n_layers=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = nn.ModuleList()
        self.mlp1 = nn.ModuleList()
        self.attn2 = nn.ModuleList()
        self.mlp2 = nn.ModuleList()
        for i in range(n_layers):
            self.attn1.append(SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop))
            self.mlp1.append(nn.Sequential(
                nn.Linear(n_embd, block_exp * n_embd),
                nn.ReLU(True),  # changed from GELU
                nn.Linear(block_exp * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            ))

            self.attn2.append(SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop))
            self.mlp2.append(nn.Sequential(
                nn.Linear(n_embd, block_exp * n_embd),
                nn.ReLU(True), # changed from GELU
                nn.Linear(block_exp * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            ))

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

        fusion_x2 = x2
        fusion_x1 = x1
        for i in range(self.n_layers):
            # x2融合到x1
            fusion_x2 = fusion_x2 + self.attn2[i](fusion_x1,fusion_x2)
            fusion_x2 = fusion_x2 + self.mlp2[i](self.ln1(fusion_x2))

            # x1融合到x2
            fusion_x1 = fusion_x1 + self.attn1[i](fusion_x2, fusion_x1)
            fusion_x1 = fusion_x1 + self.mlp1[i](self.ln2(fusion_x1))

        if T==raw_h*raw_w:
            fusion_x2 = fusion_x2.transpose(1,2).view(B,C,raw_h, raw_w)
            fusion_x1 = fusion_x1.transpose(1, 2).view(B, C, raw_h, raw_w)
        else:
            fusion_x2 = fusion_x2.transpose(1, 2).view(B, C, int(raw_h/self.patch_size[0]), int(raw_w/self.patch_size[1]))
            fusion_x2 = F.interpolate(fusion_x2, scale_factor=(self.patch_size[0], self.patch_size[1]), mode='bilinear',align_corners=True)

            fusion_x1 = fusion_x1.transpose(1, 2).view(B, C, int(raw_h/self.patch_size[0]), int(raw_w/self.patch_size[1]))
            fusion_x1 = F.interpolate(fusion_x1, scale_factor=(self.patch_size[0], self.patch_size[1]), mode='bilinear',align_corners=True)

        return input1 + fusion_x2, input2 + fusion_x1


class cross_attention_Unet(nn.Module):
    def __init__(self, input_shape1, input_shape2, num_classes):
        super(cross_attention_Unet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        # flair and t2
        self.first = nn.Sequential(nn.Conv2d(input_shape1[1], filters[0], 3, 1, 1))

        self.conv1 = res_conv_bn(filters[0], filters[0])
        self.conv2 = res_conv_bn(filters[1], filters[1])
        self.conv3 = res_conv_bn(filters[2], filters[2])
        self.conv4 = res_conv_bn(filters[3], filters[3])
        self.conv5 = res_conv_bn(filters[4], filters[4])

        self.pool1 = self.downconv(filters[0], filters[1])
        self.pool2 = self.downconv(filters[1], filters[2])
        self.pool3 = self.downconv(filters[2], filters[3])
        self.pool4 = self.downconv(filters[3], filters[4])

        self.upconv1 = self.upconv(filters[1], filters[0])
        self.upconv2 = self.upconv(filters[2], filters[1])
        self.upconv3 = self.upconv(filters[3], filters[2])
        self.upconv4 = self.upconv(filters[4], filters[3])

        self.conv8 = res_conv_bn(filters[4], filters[3])
        self.conv9 = res_conv_bn(filters[3], filters[2])
        self.conv10 = res_conv_bn(filters[2], filters[1])
        self.conv11 = res_conv_bn(filters[1], filters[0])

        self.conv12 = nn.Conv2d(filters[0], num_classes, kernel_size=3, stride=1, padding=1)
        # self.softmax = nn.Softmax(dim=1)

        # t1 and t1ce
        self.modal2_first = nn.Sequential(nn.Conv2d(input_shape2[1], filters[0], 3, 1, 1))

        self.modal2_conv1 = res_conv_bn(filters[0], filters[0])
        self.modal2_conv2 = res_conv_bn(filters[1], filters[1])
        self.modal2_conv3 = res_conv_bn(filters[2], filters[2])
        self.modal2_conv4 = res_conv_bn(filters[3], filters[3])
        self.modal2_conv5 = res_conv_bn(filters[4], filters[4])

        self.modal2_pool1 = self.downconv(filters[0], filters[1])
        self.modal2_pool2 = self.downconv(filters[1], filters[2])
        self.modal2_pool3 = self.downconv(filters[2], filters[3])
        self.modal2_pool4 = self.downconv(filters[3], filters[4])

        self.modal2_upconv1 = self.upconv(filters[1], filters[0])
        self.modal2_upconv2 = self.upconv(filters[2], filters[1])
        self.modal2_upconv3 = self.upconv(filters[3], filters[2])
        self.modal2_upconv4 = self.upconv(filters[4], filters[3])


        self.modal2_conv8 = res_conv_bn(filters[4], filters[3])
        self.modal2_conv9 = res_conv_bn(filters[3], filters[2])
        self.modal2_conv10 = res_conv_bn(filters[2], filters[1])
        self.modal2_conv11 = res_conv_bn(filters[1], filters[0])

        self.modal2_conv12 = nn.Conv2d(filters[0], num_classes, kernel_size=3, stride=1, padding=1)

        self.transform_block1 = transform_Block(img_size=[88,72], patch_size=[2,2], in_chans=filters[1],
                                                n_embd=filters[1], n_head=2, block_exp=4, attn_pdrop=0.1,
                                                resid_pdrop=0.1,n_layers=1)
        self.transform_block2 = transform_Block(img_size=[44, 36], patch_size=[1, 1], in_chans=filters[2],
                                                n_embd=filters[2], n_head=2, block_exp=4, attn_pdrop=0.1,
                                                resid_pdrop=0.1,n_layers=2)
        self.transform_block3 = transform_Block(img_size=[22, 18], patch_size=[1, 1], in_chans=filters[3],
                                                n_embd=filters[3], n_head=4, block_exp=4, attn_pdrop=0.1,
                                                resid_pdrop=0.1,n_layers=4)
        self.transform_block4 = transform_Block(img_size=[11, 9], patch_size=[1, 1], in_chans=filters[4],
                                                n_embd=filters[4], n_head=4, block_exp=4, attn_pdrop=0.1,
                                                resid_pdrop=0.1,n_layers=4)

    def upconv(self, channel_in, channel_out):
        return nn.Sequential(nn.BatchNorm2d(channel_in),
                             nn.ReLU(),
                             nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2))

    def downconv(self, channel_in, channel_out):
        return nn.Sequential(nn.BatchNorm2d(channel_in),
                             nn.ReLU(),
                             nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1))

    def forward(self, input1, input2):
        modal1_x = self.first(input1)
        modal1_x1 = self.conv1(modal1_x)
        modal1_x2 = self.pool1(modal1_x1)
        modal2_x = self.modal2_first(input2)
        modal2_x1 = self.modal2_conv1(modal2_x)
        modal2_x2 = self.modal2_pool1(modal2_x1)
        modal1_x2,modal2_x2 = self.transform_block1(modal1_x2,modal2_x2)

        modal1_x2 = self.conv2(modal1_x2)
        modal1_x3 = self.pool2(modal1_x2)
        modal2_x2 = self.modal2_conv2(modal2_x2)
        modal2_x3 = self.modal2_pool2(modal2_x2)
        modal1_x3, modal2_x3 = self.transform_block2(modal1_x3, modal2_x3)

        modal1_x3 = self.conv3(modal1_x3)
        modal1_x4 = self.pool3(modal1_x3)
        modal2_x3 = self.modal2_conv3(modal2_x3)
        modal2_x4 = self.modal2_pool3(modal2_x3)
        modal1_x4, modal2_x4 = self.transform_block3(modal1_x4, modal2_x4)

        modal1_x4 = self.conv4(modal1_x4)
        modal1_x5 = self.pool4(modal1_x4)
        modal2_x4 = self.modal2_conv4(modal2_x4)
        modal2_x5 = self.modal2_pool4(modal2_x4)
        modal1_x5, modal2_x5 = self.transform_block4(modal1_x5, modal2_x5)

        modal1_x5 = self.conv5(modal1_x5)
        modal2_x5 = self.modal2_conv5(modal2_x5)

        # t1 and t1ce
        modal1_u2 = self.upconv4(modal1_x5)
        modal1_u2 = torch.cat([modal1_u2, modal1_x4], 1)
        modal1_u2 = self.conv8(modal1_u2)
        modal1_u3 = self.upconv3(modal1_u2)
        modal1_u3 = torch.cat([modal1_u3, modal1_x3], 1)
        modal1_u3 = self.conv9(modal1_u3)
        modal1_u2 = self.upconv2(modal1_u3)
        modal1_u2 = torch.cat([modal1_u2, modal1_x2], 1)
        modal1_u2 = self.conv10(modal1_u2)
        modal1_u1 = self.upconv1(modal1_u2)
        modal1_u1 = torch.cat([modal1_u1, modal1_x1], 1)
        modal1_u1 = self.conv11(modal1_u1)

        modal1_output = self.conv12(modal1_u1)

        # flair and t2
        modal2_u2 = self.modal2_upconv4(modal2_x5)
        modal2_u2 = torch.cat([modal2_u2, modal2_x4], 1)
        modal2_u2 = self.modal2_conv8(modal2_u2)
        modal2_u3 = self.modal2_upconv3(modal2_u2)
        modal2_u3 = torch.cat([modal2_u3, modal2_x3], 1)
        modal2_u3 = self.modal2_conv9(modal2_u3)
        modal2_u2 = self.modal2_upconv2(modal2_u3)
        modal2_u2 = torch.cat([modal2_u2, modal2_x2], 1)
        modal2_u2 = self.modal2_conv10(modal2_u2)
        modal2_u1 = self.modal2_upconv1(modal2_u2)
        modal2_u1 = torch.cat([modal2_u1, modal2_x1], 1)
        modal2_u1 = self.modal2_conv11(modal2_u1)

        modal2_output = self.modal2_conv12(modal2_u1)

        return modal1_output, modal2_output


if __name__ == '__main__':
    from torch.autograd import Variable

    x1 = Variable(torch.randn(1, 2, 176, 144))
    x2 = Variable(torch.randn(1, 2, 176, 144))
    net = cross_attention_Unet([1, 2, 176, 144], [1, 2, 176, 144], 2)
    _, _ = net(x1, x2)
    from thop import profile

    flops, params = profile(net, inputs=(x1, x2))
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print(params)