"""
implementation of Squeeze and Excitation block
attach SEblock to any part of the end of the model's block you want
"""

import torch.nn as nn

class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, bias=True, norm_layer=nn.BatchNorm2d, act=True):
        super(ConvNormAct,self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias),
            norm_layer(out_ch) if norm_layer != nn.Identity() else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

class SEblock(nn.Sequential):
    def __init__(self, channel, r=16):
        super(SEblock, self).__init__(
            #squeeze
            nn.AdaptiveAvgPool2d(1),

            #excitation
            ConvNormAct(channel, channel//r, 1, bias=False, norm_layer=nn.Identity), # Linear -> ReLU
            nn.Conv2d(channel//r, channel, 1, bias=False), # Linear
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out = super(SEblock, self).forward(input)
        return out + input
