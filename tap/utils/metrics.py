import torch
from accelerate import Accelerator
from torch import Tensor
from torchmetrics.classification import (
    BinaryJaccardIndex,
    JaccardIndex,
    MulticlassJaccardIndex,
)
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional.classification.jaccard import _jaccard_index_reduce

__all__ = [
    "JaccardIndex",
    "FBIoU",
    "AverageMetricWrapper",
    "AverageMetricCollection",
    "fbiou",
    "binary_jaccard_index",
    "multiclass_jaccard_index",
    "MetricCollection",
]


def to_global_multiclass(
    classes: list[list[list[int]]], categories: dict[int, dict], *tensors: list[Tensor]
) -> list[Tensor]:
    """Convert the classes of an episode to the global classes.

    Args:
        classes (list[list[list[int]]]): The classes corresponding to batch, episode and query.
        categories (dict[int, dict]): The categories of the dataset.

    Returns:
        list[Tensor]: The updated tensors.
    """
    batch_size = len(classes)
    out_tensors = [tensor.clone() for tensor in tensors]
    cats_map = {k: i + 1 for i, k in enumerate(categories.keys())}
    for i in range(batch_size):
        # assign to longest_classes the longest list in classes[i]
        longest_classes = sorted(list(set(sum(classes[i], []))))
        for j, v in enumerate(longest_classes):
            for tensor in out_tensors:
                tensor[i] = torch.where(tensor[i] == j + 1, cats_map[v], tensor[i])
    return out_tensors


class DistributedMulticlassJaccardIndex(MulticlassJaccardIndex):
    """Distributed version of the MulticlassJaccardIndex."""

    def compute(self) -> Tensor:
        """Compute metric."""
        metric = super().compute()
        # compute the IoU of class 0 in the self.confmat
        bg_iou = self.confmat[0, 0] / (self.confmat[0, 0] + self.confmat[0, 1:].sum() + self.confmat[1:, 0].sum())
        metric = (metric * self.num_classes - bg_iou) / (self.num_classes - 1)
        return metric


class DistributedBinaryJaccardIndex(BinaryJaccardIndex):
    """Distributed version of the BinaryJaccardIndex."""

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = preds.clone(), target.clone()
        preds[preds > 0] = 1
        target[target > 0] = 1
        super().update(preds, target)
