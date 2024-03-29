import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
CVPR2019 Arcface
"""
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample (embedding size)
            out_features: size of each output sample (the number of class)
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) # -cos(m)
        self.mm = math.sin(math.pi - m) * m # sin(m)*m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        #  F.linear: Applies a linear transformation to the incoming data: y = xA^T + b.
        #                 embedding_size    number of class x embedding_size 
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) # cosin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)) # sin(theta)
        phi = cosine * self.cos_m - sine * self.sin_m # cosin(theta + margin) -> cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()) # device=CONFIG['device']
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) # y를 제외한 모든 것들은 cosine similarity로 만들어줌
        output *= self.s #마지막으로 scale을 곱합

        return output
