import torch
import torch.nn.functional as F
from torch.nn import Module
from einops import rearrange

from .utils import get_reduction


class FalsePositiveLoss(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eps = 1e-6

    def __call__(self, x, target, weight_matrix=None, **kwargs):
        included_classes = torch.stack([item.unique() for item in target])
        not_included_classes = torch.ones(x.shape[:2], device=x.device)
        not_included_classes.scatter_(1, included_classes, 0)
        not_included_classes = rearrange(not_included_classes, "b c -> b c () ()")

        soft_x = F.softmax(x, dim=1)
        false_positive_loss = soft_x * not_included_classes
        false_positive_loss = false_positive_loss.sum(dim=1) / (
            not_included_classes.sum(dim=1) + self.eps
        )
        false_positive_loss = false_positive_loss.mean()
        return false_positive_loss
