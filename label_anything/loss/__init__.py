import torch.nn as nn
import torch

from label_anything.data.utils import BatchKeys
from label_anything.loss.fp import FalsePositiveLoss
from label_anything.loss.mask import MaskEmbeddingLoss

from .dice import DiceLoss
from .focal import FocalLoss
from .rmi import RMILoss
from .prompt import PromptContrastiveLoss
from .utils import get_weight_matrix_from_labels
from label_anything.utils.utils import ResultDict


LOGITS_LOSSES = {
    "focal": FocalLoss,
    "dice": DiceLoss,
    "rmi": RMILoss,
    "fp": FalsePositiveLoss
}

PROMPT_LOSSES = {
    "prompt_contrastive": PromptContrastiveLoss,
    "masks": MaskEmbeddingLoss
}


class LabelAnythingLoss(nn.Module):
    """This loss is a linear combination of the following losses:
    - FocalLoss
    - DiceLoss
    - RMILoss
    - PromptContrastiveLoss
    - MaskEmbeddingLoss
    """

    def __init__(self, components, class_weighting=None):
        super().__init__()
        self.weights = {k: v.pop("weight") for k, v in components.items()}
        self.components = nn.ModuleDict(
            [
                [k, LOGITS_LOSSES[k](**v)]
                for k, v in components.items()
                if k in LOGITS_LOSSES
            ]
        )
        self.prompt_components = nn.ModuleDict(
            [
                [k, PROMPT_LOSSES[k](**v)]
                for k, v in components.items()
                if k in PROMPT_LOSSES
            ]
        )
        if (
            set(components.keys())
            - set(self.components.keys())
            - set(self.prompt_components.keys())
        ):
            raise ValueError(
                f"Unknown loss components: {set(components.keys()) - set(self.components.keys())}"
            )
        self.class_weighting = class_weighting
        
    def logits_loss(self, logits, target):
        weight_matrix, class_weights = None, None
        if self.class_weighting:
            num_classes = logits.shape[1]
            weight_matrix, class_weights = get_weight_matrix_from_labels(
                target, num_classes
            )

        return sum(
            self.weights[k]
            * loss(
                logits, target, weight_matrix=weight_matrix, class_weights=class_weights
            )
            for k, loss in self.components.items()
        )
    
    def prompt_loss(self, result):
        return sum(
            self.weights[k]
            * loss(result)
            for k, loss in self.prompt_components.items()
        )

    def forward(self, result, target):
        if isinstance(result, torch.Tensor): # Only logits
            logits_loss = self.logits_loss(result, target)
            return logits_loss
        
        logits_loss = self.logits_loss(result[ResultDict.LOGITS], target)
        prompt_loss = self.prompt_loss(result)
        return logits_loss + prompt_loss
