from typing import Mapping
from functools import reduce

from torch import Tensor
from torchmetrics import JaccardIndex, AUROC, F1Score, ConfusionMatrix
from torchmetrics import Precision as TPrecision
from torchmetrics import Recall as TRecall
from torchmetrics import JaccardIndex as TJaccardIndex
from torchmetrics.functional.classification.roc import roc
import torch

from copy import deepcopy

# from models import ComposedOutput


class AUC(AUROC):
    def update(self, preds: Tensor, target: Tensor) -> None:
        AUROC.update(self, preds.cpu(), target.cpu())

    def get_roc(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        if not self.num_classes:
            raise ValueError(
                f"`num_classes` bas to be positive number, but got {self.num_classes}"
            )
        return roc(preds, target, self.num_classes, self.pos_label)


def PerClassAUC(name, code):
    def __init__(self, name, code, *args, **kwargs):
        AUC.__init__(self, **kwargs)
        self.code = code

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds[:, self.code, ::].flatten()
        target = (target == code).flatten()
        AUC.update(self, preds, target)

    metric = type(name, (AUC,), {"update": update, "__init__": __init__})
    return metric(name, code)


def get_multiclass(names, values):
    macro = values[~values.isnan()].mean()
    per_class = {f"{names[i]}": f1 for i, f1 in enumerate(values)}
    per_class[names[-1]] = macro
    return per_class


def remove_aux(preds):
    if isinstance(preds, tuple):
        return preds[0]
    if isinstance(preds, dict):
        return preds["out"]
    if isinstance(preds, ComposedOutput):
        return preds.main
    return preds


def remove_padding(preds, target, padding):
    for i in range(preds.shape[0]):
        w_slice = slice(0, preds.shape[2] - padding[i][1])
        h_slice = slice(0, preds.shape[3] - padding[i][0])
        pred = preds[i, :, w_slice, h_slice]
        targ = target[i, w_slice, h_slice]
        yield pred.unsqueeze(0), targ.unsqueeze(0)


class F1(F1Score):
    def __init__(self, *args, **kwargs):
        self.component_names = [f"f1_{i}" for i in range(kwargs["num_classes"])] + [
            "f1"
        ]
        super().__init__(**kwargs, average="none", mdmc_average="global")

    def update(self, preds: Tensor, target: Tensor, padding=None) -> None:
        preds = remove_aux(preds)
        if padding is not None:
            for pred, tar in remove_padding(preds, target, padding):
                super().update(pred, tar)
        super().update(preds, target)

    def compute(self):
        per_class_f1 = super().compute()
        return get_multiclass(self.component_names, per_class_f1)


class Precision(TPrecision):
    def __init__(self, *args, **kwargs):
        self.component_names = [
            f"precision_{i}" for i in range(kwargs["num_classes"])
        ] + ["precision"]
        super().__init__(**kwargs, average="none", mdmc_average="global")

    def update(self, preds: Tensor, target: Tensor, padding=None) -> None:
        preds = remove_aux(preds)
        if padding is not None:
            for pred, tar in remove_padding(preds, target, padding):
                super().update(pred, tar)
        super().update(preds, target)

    def compute(self):
        per_class_precision = super().compute()
        return get_multiclass(self.component_names, per_class_precision)


class Recall(TRecall):
    def __init__(self, *args, **kwargs):
        self.component_names = [f"recall_{i}" for i in range(kwargs["num_classes"])] + [
            "recall"
        ]
        super().__init__(**kwargs, average="none", mdmc_average="global")

    def update(self, preds: Tensor, target: Tensor, padding=None) -> None:
        preds = remove_aux(preds)
        if padding is not None:
            for pred, tar in remove_padding(preds, target, padding):
                super().update(pred, tar)
        super().update(preds, target)

    def compute(self):
        per_class_recall = super().compute()
        return get_multiclass(self.component_names, per_class_recall)


class ConfMat(ConfusionMatrix):
    def __init__(self, num_classes: int, *args, **kwargs):
        self.component_names = [
            f"cf_{i}_{j}" for i in range(num_classes) for j in range(num_classes)
        ]
        super().__init__(num_classes, *args, **kwargs)

    # def update(self, preds: Tensor, target: Tensor) -> None:
    #     return super().update(preds.argmax(dim=1), target.argmax(dim=1))

    def compute(self) -> Tensor:
        # PlaceHolder value
        cf = super(ConfMat, self).compute()
        names = [
            f"cf_{i}_{j}"
            for i in range(self.num_classes)
            for j in range(self.num_classes)
        ]
        return dict(zip(names, map(lambda x: x.item(), cf.flatten())))

    def update(self, preds: Tensor, target: Tensor, padding=None) -> None:
        preds = remove_aux(preds)
        if padding is not None:
            for pred, tar in remove_padding(preds, target, padding):
                super().update(pred, tar)
        super().update(preds, target)

    def get_cf(self):
        return super().compute()


def metric_instance(name: str, params: dict) -> dict:
    if params.get("discriminator") is not None:
        params = deepcopy(params)
        names = params.pop("discriminator")
        return {
            subname: METRICS[name](subname, code, **params) for subname, code in names
        }
    return {name: METRICS[name](**params)}


def metrics_factory(metrics_params: Mapping) -> dict:
    return reduce(
        lambda a, b: {**a, **b},
        [metric_instance(name, params) for name, params in metrics_params.items()],
    )


def get_metric_titles_components_mapping(metrics):
    m_dict = {
        metric_name: metric.component_names
        if hasattr(metric, "component_names")
        else [metric_name]
        for metric_name, metric in metrics.items()
    }
    new_dict = {}

    for key, values in m_dict.items():
        for value in values:
            new_dict[value] = key
    return new_dict


METRICS = {
    "jaccard": JaccardIndex,
    "auc": AUC,
    "perclassauc": PerClassAUC,
    "f1": F1,
    "f1score": F1Score,
    "precision": Precision,
    "recall": Recall,
    "conf_mat": ConfMat,
}
