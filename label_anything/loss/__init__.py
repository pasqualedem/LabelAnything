import torch.nn as nn
import torch

from label_anything.data.utils import BatchKeys
from label_anything.loss.fp import FalsePositiveLoss
from label_anything.loss.mask import MaskEmbeddingLoss

from .dice import DiceLoss
from .focal import FocalLoss
from .rmi import RMILoss
from .prompt import ClassEmbeddingContrastiveLoss, PromptContrastiveLoss
from .utils import get_weight_matrix_from_labels
from label_anything.utils.utils import ResultDict, LossDict


LOGITS_LOSSES = {
    "focal": FocalLoss,
    "dice": DiceLoss,
    "rmi": RMILoss,
    "fp": FalsePositiveLoss,
}

PROMPT_LOSSES = {
    "prompt_contrastive": PromptContrastiveLoss,
    "emb_contrastive": ClassEmbeddingContrastiveLoss,
    "masks": MaskEmbeddingLoss,
}


class LabelAnythingLoss(nn.Module):
    """This loss is a linear combination of the following losses:
    - FocalLoss
    - DiceLoss
    - RMILoss
    - PromptContrastiveLoss
    - MaskEmbeddingLoss
    - ClassEmbeddingContrastiveLoss
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

        loss_values = []
        loss_dict = {}
        for k, loss in self.components.items():
            loss_res = self.weights[k] * loss(
                logits, target, weight_matrix=weight_matrix, class_weights=class_weights
            )
            if isinstance(loss_res, dict):
                loss_value = loss_res[LossDict.VALUE]
                loss_dict = {**loss_dict, **loss_res[LossDict.COMPONENTS]}
            else:
                loss_dict[k] = loss_res.item() if isinstance(loss_res, torch.Tensor) else loss_res
                loss_value = loss_res
            loss_values.append(self.weights[k] * loss_value)
            loss_value = sum(loss_values)
        return {LossDict.VALUE: loss_value, LossDict.COMPONENTS: loss_dict}

    def prompt_loss(self, result):
        loss_values = []
        loss_dict = {}
        for k, loss in self.prompt_components.items():
            loss_res = loss(result)
            if isinstance(loss_res, dict):
                loss_value = loss_res[LossDict.VALUE]
                loss_dict = {**loss_dict, **loss_res[LossDict.COMPONENTS]}
            else:
                loss_dict[k] = loss_res.item() if isinstance(loss_res, torch.Tensor) else loss_res
                loss_value = loss_res
            loss_values.append(self.weights[k] * loss_value)
        loss_value = sum(loss_values)
        return {LossDict.VALUE: loss_value, LossDict.COMPONENTS: loss_dict}

    def forward(self, result, target):
        if isinstance(result, torch.Tensor):  # Only logits
            logits_loss = self.logits_loss(result, target)
            return logits_loss

        logits_loss = self.logits_loss(result[ResultDict.LOGITS], target)
        prompt_loss = self.prompt_loss(result)
        return {
            LossDict.VALUE: logits_loss[LossDict.VALUE] + prompt_loss[LossDict.VALUE],
            LossDict.COMPONENTS: {**logits_loss[LossDict.COMPONENTS], **prompt_loss[LossDict.COMPONENTS]}
        }
