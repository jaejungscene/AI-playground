class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

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
      
  
  
# parameters 100k under, simple mobilenetV3
  class JaeJungNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(JaeJungNet ,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
        )# 32

        self.layer2 = nn.Sequential(
            # kernel_size, in_size, expand_size, out_size, nolinear, seblock, stride
            InvertedResidual(3, 16, 16, 16, nn.ReLU(inplace=True), SEblock(16), 2), # 16
            InvertedResidual(3, 16, 64, 24, nn.ReLU(inplace=True), SEblock(24), 1),
            InvertedResidual(3, 24, 88, 40, nn.ReLU(inplace=True), SEblock(40), 1),
            InvertedResidual(3, 40, 96, 40, hswish(), SEblock(40), 2),
            InvertedResidual(3, 40, 120, 48, hswish(), SEblock(48), 1),
            InvertedResidual(3, 48, 288, 96, hswish(), SEblock(96), 1),
        )

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(96, num_classes)
        ) 
        
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d): # kaiming he initialization
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
