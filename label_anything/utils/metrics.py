import torch
from torch import Tensor
from torchmetrics.classification import BinaryJaccardIndex, JaccardIndex
from torchmetrics import MetricCollection
from torchmetrics.functional.classification import binary_jaccard_index, multiclass_jaccard_index

from label_anything.utils.utils import RunningAverage

__all__ = [
    "JaccardIndex",
    "FBIoU",
    "AverageMetricWrapper",
    "AverageMetricCollection",
    "fbiou",
    "binary_jaccard_index",
    "multiclass_jaccard_index"
    "MetricCollection"
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

