import torch
import torch.nn as nn
from collections import OrderedDict


model_cfg = {
    "densenet121":[6,12,24,16],
    "densenet169":[6,12,32,32],
    "densenet201":[6,12,48,32],
    "densenet264":[6,12,64,48],
}


class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, bottleneck_size,
    norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU) -> None:
        super(DenseLayer, self).__init__()
        self.add_module("norm1", norm_layer(in_ch))
        self.add_module("act1", act_layer(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_ch, growth_rate*bottleneck_size, 1, 1, bias=False))
        self.add_module("norm2", norm_layer(growth_rate*bottleneck_size))
        self.add_module("act2", act_layer(inplace=True))
        self.add_module("conv2", nn.Conv2d(growth_rate*bottleneck_size, growth_rate, 3, 1, 1, bias=False))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x = [x]
        x = torch.cat(x, 1)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x



class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layer, in_ch, growth_rate, bottleneck_size,
    norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU) -> None:
        super(DenseBlock, self).__init__()
        for i in range(num_layer):
            layer = DenseLayer(
                in_ch=in_ch+(i*growth_rate),
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            self.add_module(f"denselayer{i+1}", layer)

    def forward(self, init_features:torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        features = torch.cat(features, 1)
        return features



class TransitionLayer(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU
    ) -> None:
        super(TransitionLayer, self).__init__()
        self.add_module("norm", norm_layer(in_ch))
        self.add_module("relu", act_layer(inplace=True))
        self.add_module("conv", nn.Conv2d(in_ch, out_ch, 1, 1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, 2))
    
    def forward(self, x):
        return self.pool(self.conv(self.relu(self.norm(x))))



class DenseNet(nn.Module):
    """
    hyperparameter for Densnet:
        growth_rate - how many filters to add each layer ('k' in paper)
        bottleneck_size - multiplicative factor at the bottle neck layer
    """
    def __init__(self, block_cfg, growth_rate=32, bottleneck_size=4, num_classes=1000,
    norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU) -> None:
        super(DenseNet, self).__init__()
        init_features = growth_rate * 2
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, init_features, 7, 2, 3, bias=False)),
            ("norm0", norm_layer(init_features)),
            ("relu0", act_layer(init_features)),
            ("pool0", nn.MaxPool2d(3, 2, 1)),
        ]))

        for i, num_layer in enumerate(block_cfg):
            block = DenseBlock(
                num_layer=num_layer,
                in_ch=init_features,
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            self.features.add_module(f"denseblock{i+1}", block)
            init_features = init_features + (growth_rate * num_layer)
            if i+1 != len(block_cfg):
                transition = TransitionLayer(
                    in_ch=init_features, out_ch=init_features//2,
                    norm_layer=norm_layer, act_layer=act_layer
                )
                self.features.add_module(f"transition{i+1}", transition)
                init_features = init_features//2

        self.features.add_module("norm5", norm_layer(init_features))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(init_features, num_classes)
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        x = self.features(x)
        x = self.global_pool(x).flatten(1, -1)
        return self.classifier(x)