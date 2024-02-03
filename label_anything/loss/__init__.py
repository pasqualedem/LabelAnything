import torch.nn as nn

from label_anything.data.utils import BatchKeys

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
}

PROMPT_LOSSES = {
    "prompt_contrastive": PromptContrastiveLoss,
}


class LabelAnythingLoss(nn.Module):
    """This loss is a linear combination of the following losses:
    - FocalLoss
    - DiceLoss
    - RMILoss
    - PromptContrastiveLoss
    """

    def __init__(self, components, class_weighting=None):
        super().__init__()
        self.weights = {k: v.pop("weight") for k, v in components.items()}
        self.components = {
            k: LOGITS_LOSSES[k](**v)
            for k, v in components.items()
            if k in LOGITS_LOSSES
        }
        self.prompt_components = {
            k: PROMPT_LOSSES[k](**v)
            for k, v in components.items()
            if k in PROMPT_LOSSES
        }
        if (
            set(components.keys())
            - set(self.components.keys())
            - set(self.prompt_components.keys())
        ):
            raise ValueError(
                f"Unknown loss components: {set(components.keys()) - set(self.components.keys())}"
            )
        self.class_weighting = class_weighting

    def forward(self, result, target):
        logits = result[ResultDict.LOGITS]

        weight_matrix, class_weights = None, None
        if self.class_weighting:
            num_classes = logits.shape[1]
            weight_matrix, class_weights = get_weight_matrix_from_labels(
                target, num_classes
            )

        logits_loss = sum(
            self.weights[k]
            * loss(
                logits, target, weight_matrix=weight_matrix, class_weights=class_weights
            )
            for k, loss in self.components.items()
        )
        prompt_loss = sum(
            self.weights[k] * loss(
                result[ResultDict.EXAMPLES_CLASS_EMBS],
                result[BatchKeys.FLAG_MASKS],
                result[BatchKeys.FLAG_POINTS],
                result[BatchKeys.FLAG_BBOXES]
                )
            for k, loss in self.prompt_components.items()
        )
        return logits_loss + prompt_loss
