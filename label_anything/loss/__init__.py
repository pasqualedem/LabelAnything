from .dice import DiceLoss
from .focal import FocalLoss
from .rmi import RMILoss

import torch.nn as nn


LOSSES = {
    "focal": FocalLoss,
    "dice": DiceLoss,
    "rmi": RMILoss,

}


class LabelAnythingLoss(nn.Module):
    """This loss is a linear combination of the following losses:
    - FocalLoss
    - DiceLoss
    - RMILoss
    """
    def __init__(self, components):
        super().__init__()
        params = {
            k: v["params"] for k, v in components.items()
        }
        self.weights = {
            k: v["weight"] for k, v in components.items()
        }
        self.components = {
            k: LOSSES[k](**v) for k, v in params.items()
        }

    def forward(self, logits, target):
        return sum(
            w * loss(logits, target)
            for w, loss in zip(self.weights.values(), self.components.values())
        )