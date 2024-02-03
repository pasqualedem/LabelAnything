import lovely_tensors as lt

lt.monkey_patch()
import torch
from accelerate import Accelerator
from torch import Tensor
from torchmetrics.classification import (
    BinaryJaccardIndex,
    JaccardIndex,
    MulticlassJaccardIndex,
)
from torchmetrics.functional.classification import binary_jaccard_index

from label_anything.utils.utils import RunningAverage

__all__ = [
    "JaccardIndex",
    "FBIoU",
    "AverageMetricWrapper",
    "AverageMetricCollection",
    "fbiou",
    "binary_jaccard_index",
    "multiclass_jaccard_index" "MetricCollection",
]


class FBIoU(BinaryJaccardIndex):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update((preds != 0).int(), (target != 0).int())


class AverageMetricWrapper(RunningAverage):
    def __init__(self, metric_fn, accelerator):
        super().__init__()
        self.accelerator = accelerator
        self.metric_fn = metric_fn

    def update(
        self, preds: Tensor, target: Tensor, num_classes: int, ignore_index: int = -100
    ) -> None:
        value = torch.mean(
            self.accelerator.gather(
                self.metric_fn(
                    preds=preds,
                    target=target,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
            )
        ).item()
        super().update(value)
        return value

    def compute(self):
        return super().compute()


class AverageMetricCollection:
    def __init__(self, prefix: str, metrics: dict, accelerator):
        self.prefix = prefix
        self.metrics = metrics
        self.accelerator = accelerator
        self.metric_names = list(metrics.keys())
        self.metrics = {
            k: AverageMetricWrapper(v, accelerator) for k, v in metrics.items()
        }

    def update(
        self, preds: Tensor, target: Tensor, num_classes, ignore_index: int = -100
    ) -> None:
        metric_values = {}
        for metric_name, metric in self.metrics.items():
            metric_values[self.prefix + metric_name] = metric.update(
                preds, target, num_classes, ignore_index
            )
        return metric_values

    def compute(self):
        return {self.prefix + k: v.compute() for k, v in self.metrics.items()}


def fbiou(preds: Tensor, target: Tensor, ignore_index=-100, *args, **kwargs) -> None:
    # target is (B, H, W) while preds is (B, C, H, W)
    target = target.clone()
    # collapse pred to (B, H, W) (foreground/background)
    preds = preds.clone()
    preds[preds != 0] = 1
    target[target != 0] = 1
    return binary_jaccard_index(preds, target, ignore_index=ignore_index)


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
        longest_classes = max(classes[i], key=len)
        for j, v in enumerate(longest_classes):
            for tensor in out_tensors:
                tensor[i] = torch.where(tensor[i] == j + 1, cats_map[v], tensor[i])
    return out_tensors


class DistributedMulticlassJaccardIndex(MulticlassJaccardIndex):
    """Distributed version of the MulticlassJaccardIndex.

    Please, use a value of `ignore_index` that is >= 0.
    """

    def __init__(self, accelerator: Accelerator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator = accelerator

    def update(self, preds: Tensor, target: Tensor) -> None:
        print("preds", preds)
        print("target", target)
        # put all 0s of target to ignore_index
        target = target.where(target == 0, self.ignore_index)
        preds = preds.clone()
        # subtract 1 from values of target and preds, if they are greater than 0
        target[target > 0] -= 1
        preds[preds > 0] -= 1
        super().update(preds, target)

    def compute(self) -> Tensor:
        self.confmat = self.accelerator.gather(self.confmat)
        return super().compute()


class DistributedBinaryJaccardIndex(BinaryJaccardIndex):
    """Distributed version of the BinaryJaccardIndex."""

    def __init__(self, accelerator: Accelerator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator = accelerator

    def update(self, preds: Tensor, target: Tensor) -> None:
        print("preds", preds)
        print("target", target)
        preds, target = preds.clone(), target.clone()
        preds[preds > 0] = 1
        target[target > 0] = 1
        super().update(preds, target)

    def compute(self) -> Tensor:
        self.confmat = self.accelerator.gather(self.confmat)
        return super().compute()
