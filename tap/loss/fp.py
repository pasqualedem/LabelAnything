import torch
import torch.nn.functional as F
from torch.nn import Module
from einops import rearrange

from .utils import get_reduction


class FalsePositiveLoss(Module):
    def __init__(self, ignore_index=-100, **kwargs):
        super().__init__()
        self.eps = 1e-6
        self.ignore_index = ignore_index

    def __call__(self, x, target, weight_matrix=None, **kwargs):
        mask = (target != self.ignore_index)
        valid_elements = mask.sum()
        
        full_target = target.clone()
        full_target[~mask] = 0
        
        included_classes = [item.unique() for item in full_target]
        not_included_classes = torch.ones(x.shape[:2], device=x.device)
        for i, item in enumerate(included_classes):
            not_included_classes[i].scatter_(0, item, 0)
        not_included_classes = rearrange(not_included_classes, "b c -> b c () ()")
        mask = rearrange(mask, "b h w -> b () h w")

        soft_x = F.softmax(x, dim=1)
        false_positive_loss = soft_x * not_included_classes * mask
        false_positive_loss = false_positive_loss.sum(dim=1) / (
            not_included_classes.sum(dim=1) + self.eps
        )
        false_positive_loss = false_positive_loss.sum() / valid_elements
        return false_positive_loss
