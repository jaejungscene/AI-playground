import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class ConvNormAct(nn.Sequential):
    def __init__(self, inch, outch, kernel, stride=1, padding=0, groups=1, bias=True, norm=True, act=True):
        super(ConvNormAct,self).__init__(
            nn.Conv2d(inch, outch, kernel, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(outch) if norm else nn.Identity(),
            nn.ReLU() if act else nn.Identity(),
        )


class SEblock(nn.Sequential):
    def __init__(self, channel, r=16):
        super(SEblock, self).__init__(
            #squeeze
            nn.AdaptiveAvgPool2d(1),
            #excitation
            ConvNormAct(channel, channel//r, 1, bias=False), # Linear -> ReLU
            nn.Conv2d(channel//r, channel, 1, bias=False), # Linear
            hsigmoid()
        )

class InvertedResidual(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.se = semodule

        # stardard conv
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        # Depthwise conv
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        # Pointwise conv
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out) + out
        out = out + self.shortcut(x) if self.stride==1 else out
        return out