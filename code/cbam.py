"""
implementation of CBAM(Convolutional Block Attention Module)
attach CBAM module to any part of the end of the model's block you want
"""

import torch
import torch.nn as nn

class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, bias=True, norm_layer=nn.BatchNorm2d, act=True):
        super(ConvNormAct,self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias),
            norm_layer(out_ch) if norm_layer != nn.Identity() else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

class CBAM(nn.Module):
    def __init__(self, channel, r=16):
        super(CBAM, self).__init__()
        self.avg_channel = nn.AdaptiveAvgPool2d(1)
        self.max_channel = nn.AdaptiveMaxPool2d(1)
        self.shared_excitation = nn.Sequential(
            ConvNormAct(channel, channel//r, 1, bias=False, norm_layer=nn.Identity),
            nn.Conv2d(channel//r, channel, 1, bias=False)
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=7//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        ## channel attention ##
        out1 = self.avg_channel(input)
        out1 = self.shared_excitation(out1)
        out2 = self.max_channel(input)
        out2 = self.shared_excitation(out2)
        channel_attention = nn.Sigmoid()(out1+out2) # (batch, channel, 1, 1)
        input = input * channel_attention

        ## spatial attention ##
        batch, size,_,_ = input.shape
        avg_spatial = input.mean(dim=1).reshape(batch, 1, size, -1) # (batch, 1, H, W)
        max_spatial = input.max(dim=1)[0].reshape(batch, 1, size, -1) # (batch, 1, H, W)
        spatial_attention = torch.cat([avg_spatial, max_spatial], 1)
        spatial_attention = self.conv_spatial(spatial_attention)
        input = input * spatial_attention

        return input
