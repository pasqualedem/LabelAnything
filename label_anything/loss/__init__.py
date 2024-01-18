from .dice import DiceLoss
from .focal import FocalLoss
from .rmi import RMILoss
from .utils import get_weight_matrix_from_labels

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

    def __init__(self, components, class_weighting=None):
        super().__init__()
        self.weights = {k: v.pop("weight") for k, v in components.items()}
        self.components = {k: LOSSES[k](**v) for k, v in components.items()}
        self.class_weighting = class_weighting

    def forward(self, logits, target):
        weight_matrix, class_weights = None, None
        if self.class_weighting:
            num_classes = logits.shape[1]
            weight_matrix, class_weights = get_weight_matrix_from_labels(target, num_classes)

        return sum(
            w
            * loss(
                logits, target, weight_matrix=weight_matrix, class_weights=class_weights
            )
            for w, loss in zip(self.weights.values(), self.components.values())
        )
