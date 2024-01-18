import torch
import torch.nn.functional as F
from torch.nn import Module
from label_anything.utils.utils import substitute_values
from .utils import get_reduction


class FocalLoss(Module):
    def __init__(
        self, gamma: float = 2.0, reduction: str = "mean", **kwargs
    ):
        super().__init__()
        self.gamma = gamma

        self.reduction = get_reduction(reduction)

    def __call__(self, x, target, weight_matrix=None, **kwargs):
        ce_loss = F.cross_entropy(x, target, reduction="none")
        pt = torch.exp(-ce_loss)
        if weight_matrix is not None:
            focal_loss = torch.pow((1 - pt), self.gamma) * weight_matrix * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return self.reduction(focal_loss)