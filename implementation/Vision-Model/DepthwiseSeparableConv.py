import torch.nn as nn

class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, inch, outch, kernel, stride=1, padding=0):
        super(DepthwiseSeparableConv,self).__init__(
            nn.Conv2d(inch, inch, 3, stride, padding, groups=inch, bias=False),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),

            nn.Conv2d(inch, outch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True),
        )
