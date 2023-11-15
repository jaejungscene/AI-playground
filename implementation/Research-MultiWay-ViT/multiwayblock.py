import torch
import torch.nn as nn
import numpy as np


class MultiWayBlock(nn.Module):
    r"""
    num_path: a factor that determines how many inception paths are created.
        so, num_path 1 means gerneral attention.
    direction: if direction is 0, path is odd and there are both pixshuf & unpixshuf
               if direction is 1, there are only unpixshuf path (channel up, spatial size down)
               if direction is 2, there are only pixshuf path (channel down, spatial size up)
    """
    def __init__(self, config, block,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super(MultiWayBlock, self).__init__()
        # assert num_path%2==1, f"num_path must be odd number"
        assert config.direction==0 or config.direction==1 or config.direction==2, "direction must be 0 or 1 or 2."
        if (config.direction==0 and config.num_path%2==0):
            assert False, "if direction is 0, num_path must be odd number."
        
        self.dim = config.hidden_size
        self.num_path = config.num_path
        self.median = (self.num_path//2)+1 if self.num_path%2==1 else self.num_path//2
        self.total_dim = self.dim * self.num_path
        self.pixshuf_factor = config.pixshuf_factor
        self.direction = config.direction
        self.concat = config.concat
        # self.pos_embed = config.pos_embed
        self.make_path(config, block)
        if self.concat:
            self.norm = norm_layer(self.total_dim)
            self.fc = nn.Linear(self.total_dim, self.dim)
    

    def make_path(self, config, block)->None:
        for i in range(1, self.num_path+1):
            # print(self.median)
            if self.direction == 0:
                if i < self.median:
                    new_dim = self.dim * (self.pixshuf_factor**(i*2))
                elif i > self.median:   
                    new_dim = int(self.dim / (self.pixshuf_factor**((i-self.median)*2)))
                else:   new_dim = self.dim
            elif self.direction == 1:
                if i==1:    new_dim = self.dim
                else:   new_dim = self.dim * (self.pixshuf_factor**((i-1)*2))
            else:
                if i==1:    new_dim = self.dim
                else:   new_dim = int(self.dim / (self.pixshuf_factor**((i-1)*2)))
            # print(new_dim)
            config.hidden_size = new_dim
            # print(">>>>>>>> ",config.hidden_size)
            self.add_module(f"path{i}", block(config, False))
            # self.total_dim = self.total_dim + new_dim
        # raise Exception("----------end-----------")
        config.hidden_size = self.dim

    def _forward_each_paths(self, x:torch.Tensor) -> torch.Tensor:
        features = []
        flag = True
        for i in range(1, self.num_path+1):
            if self.direction == 0:
                if i < self.median:
                    shuf1 = nn.PixelUnshuffle(self.pixshuf_factor*i)
                    shuf2 = nn.PixelShuffle(self.pixshuf_factor*i)
                elif i > self.median:
                    shuf1 = nn.PixelShuffle(self.pixshuf_factor*(i-self.median))
                    shuf2 = nn.PixelUnshuffle(self.pixshuf_factor*(i-self.median))
                else:
                    flag = False
            elif self.direction == 1: # only up channel
                if i==1:
                    flag = False
                else:   
                    shuf1 = nn.PixelUnshuffle(self.pixshuf_factor*(i-1))
                    shuf2 = nn.PixelShuffle(self.pixshuf_factor*(i-1))
            else: # only down channel 
                if i==1:
                    flag = False
                else:   
                    shuf1 = nn.PixelShuffle(self.pixshuf_factor*(i-1))
                    shuf2 = nn.PixelUnshuffle(self.pixshuf_factor*(i-1))

            x = shuf1(x) if flag else x
            B, D, H, W = x.shape
            x = x.contiguous().view(B,D,H*W).permute(0,2,1)
            x, attn_probs = getattr(self, f"path{i}")(x)
            x = x.permute(0,2,1).view(B,D,H,W)
            x = shuf2(x) if flag else x

            features.append(x)
            flag = True
        return features, attn_probs

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = W = int(np.sqrt(N))
        x = x.permute(0,2,1).view(B,D,H,W)
        features, attn_probs = self._forward_each_paths(x)
        if self.concat:
            x = torch.cat(features,dim=1)
            x = x.view(B,self.total_dim,N).permute(0,2,1)
            x = self.fc(self.norm(x))
        else:
            x = features[0]
            for f in features[1:]:
                x = x + f
            x = x.view(B,self.dim,N).permute(0,2,1)
        
        return x, attn_probs