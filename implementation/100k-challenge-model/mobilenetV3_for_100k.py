import torch.nn as nn
from torch.nn import init
from mobilenetV3_for_100k import InvertedResidual, SEblock, hswish
  
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
