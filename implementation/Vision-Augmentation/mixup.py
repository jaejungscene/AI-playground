import torch
import random
import torch.nn.functional as F

class MixUP:
    def __init__(self, p=0.5, alpha=1.0, nclass=1000):
        self.p = p
        self.alpha = alpha
        self.nclass = nclass

    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.nclass).to(dtype=batch.dtype)

        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        batch = batch * (1-ratio) + batch_roll * ratio
        target = target * (1-ratio) + target_roll * ratio

        return batch, target
