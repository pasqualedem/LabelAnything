import torch
import numpy as np
from accelerate import Accelerator
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryJaccardIndex,
    JaccardIndex,
    MulticlassJaccardIndex,
)
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional.classification.jaccard import _jaccard_index_reduce

from label_anything.utils.utils import substitute_values

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


class StrictMeanIoU(MulticlassJaccardIndex):
    """Distributed version of the MulticlassJaccardIndex."""

    def compute(self) -> Tensor:
        """Compute metric."""
        metric = super().compute()
        # compute the IoU of class 0 in the self.confmat
        bg_iou = self.confmat[0, 0] / (self.confmat[0, 0] + self.confmat[0, 1:].sum() + self.confmat[1:, 0].sum())
        metric = (metric * self.num_classes - bg_iou) / (self.num_classes - 1)
        return metric
    
    
class MeanIoU(MulticlassJaccardIndex):
    pass


class DistributedBinaryJaccardIndex(BinaryJaccardIndex):
    """Distributed version of the BinaryJaccardIndex."""

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = preds.clone(), target.clone()
        preds[preds > 0] = 1
        target[target > 0] = 1
        super().update(preds, target)


class PmIoU(Metric):
    """
    Compute mean IoU (PaNet implementation)

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_runs:
            number of test runs
    """
    def __init__(self, max_label=20, n_runs=None):
        super().__init__()
        self.labels = list(range(max_label + 1))  # all class labels
        self.n_runs = 1 if n_runs is None else n_runs

        # list of list of array, each array save the TP/FP/FN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_runs)]
        self.fp_lst = [[] for _ in range(self.n_runs)]
        self.fn_lst = [[] for _ in range(self.n_runs)]

    def update(self, pred, target, labels=None, n_run=None):
        """
        update the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args:
            pred:
                predicted mask array, expected shape is H x W
            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        assert pred.shape == target.shape
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        if len(pred.shape) == 3:
            if pred.shape[0] != 1:
                for i in range(pred.shape[0]):
                    self.update(pred[i], target[i])
            else:
                pred = pred[0]
                target = target[0]

        if self.n_runs == 1:
            n_run = 0

        # array to save the TP/FP/FN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)

        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            idx = np.where(np.logical_and(pred == j, target != 255))
            pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # Get the location of the pixels that are class j in ground truth
            idx = np.where(target == j)
            target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

            if target_idx_j:  # if ground-truth contains this class
                tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))
                fp_arr[label] = len(pred_idx_j - target_idx_j)
                fn_arr[label] = len(target_idx_j - pred_idx_j)

        self.tp_lst[n_run].append(tp_arr)
        self.fp_lst[n_run].append(fp_arr)
        self.fn_lst[n_run].append(fn_arr)

    def compute(self, labels=None, n_run=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels[1:]  # exclude BG
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely
            # Average across n_runs, then average over classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            # return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
            #         mIoU.mean(axis=0), mIoU.std(axis=0))
            return torch.tensor(mIoU.mean(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            # return mIoU_class, mIoU
            return mIoU

    def get_mIoU_binary(self, n_run=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0)
                      for run in range(self.n_runs)]

            # Sum over all foreground classes
            tp_sum = [np.c_[tp_sum[run][0], np.nansum(tp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fp_sum = [np.c_[fp_sum[run][0], np.nansum(fp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fn_sum = [np.c_[fn_sum[run][0], np.nansum(fn_sum[run][1:])]
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely and average across classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0)

            # Sum over all foreground classes
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

class DmIoU(Metric):
    """
    Compute mean IoU (DENet implementation)
    """
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes + 1
        self.mat = None
        self.cls_iou = None

    def update(self, label_preds, label_trues):
        label_trues = label_trues.flatten().cpu().numpy()
        label_preds = label_preds.flatten().cpu().numpy()
        n = self.num_classes
        if self.mat is None:
            self.mat = np.zeros((n, n))
        k = (label_trues >= 0) & (label_trues < n)
        inds = n * label_trues[k].astype(int) + label_preds[k]
        self.mat += np.bincount(inds, minlength=n ** 2).reshape(n, n)

    def compute(self, eps=1e-8):
        hist = self.mat.astype(float)
        numerator = np.diag(hist)
        denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        denominator = np.clip(denominator, eps, np.inf)
        iu = numerator / denominator
        self.cls_iou = {cls_id: iou for cls_id, iou in enumerate(iu)}
        return torch.tensor(np.nanmean(iu))

    def class_iou(self):
        if self.cls_iou is None:
            self.mean_iou()
        return self.cls_iou

    def mean_subclasses_iou(self, subclasses):
        mean = [v if k in subclasses else 0 for k, v in self.class_iou().items()]
        return sum(mean) / len(mean)

    def subclasses_iou(self, subclasses):
        return {k: v for k, v in self.class_iou().items() if k in subclasses}
    
    
class ImIoU(Metric):
    """
    Compute mean IoU (ASNet implementation)
    """
    def __init__(self, class_ids, n_ways=2, ignore_index=255, benchmark='pascal'):
        super().__init__()
        self.benchmark = benchmark
        self.way = n_ways
        self.class_ids_interest = torch.tensor(class_ids)
        self.ignore_index = ignore_index

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80

        self.total_area_inter = torch.zeros((self.nclass + 1, ), dtype=torch.float32)
        self.total_area_union = torch.zeros((self.nclass + 1, ), dtype=torch.float32)
        self.ones = torch.ones((len(self.class_ids_interest), ), dtype=torch.float32)

        self.seg_loss_sum = 0.
        self.seg_loss_count = 0.

        self.cls_loss_sum = 0.
        self.cls_er_sum = 0.
        self.cls_loss_count = 0.
        self.cls_er_count = 0.

    def update(self, pred_mask, gt_mask, loss=None):
        # ignore_mask = batch.get('query_ignore_idx')
        ignore_mask = None
        # gt_mask = batch.get('query_mask')
        # support_classes = batch.get('support_classes')
        support_classes = sorted((set(pred_mask.unique().tolist()).union(set(gt_mask.unique().tolist())) - {self.ignore_index}) - {0})
        support_classes = torch.tensor(list(support_classes)).to(pred_mask.device).unsqueeze(0)
        assert pred_mask.shape[0] == 1, "This implementation of ImIoU only supports batch size of 1"

        if ignore_mask is not None:
            pred_mask[ignore_mask == self.ignore_index] = self.ignore_index
            gt_mask[ignore_mask == self.ignore_index] = self.ignore_index

        pred_mask, gt_mask, support_classes = pred_mask.cpu(), gt_mask.cpu(), support_classes.cpu()
        class_dicts = self.return_class_mapping_dict(support_classes)

        uq = class_dicts[0]
        values = torch.arange(len(uq))
        pred_mask = substitute_values(pred_mask, values, unique=uq)
        gt_mask = substitute_values(gt_mask, values, unique=uq)

        samplewise_iou = []  # samplewise iou is for visualization purpose only
        for class_dict, pred_mask_i, gt_mask_i in zip(class_dicts, pred_mask, gt_mask):
            area_inter, area_union = self.intersect_and_union(pred_mask_i, gt_mask_i)

            if torch.sum(gt_mask_i.sum()) == 0:  # no foreground
                samplewise_iou.append(torch.tensor([float('nan')]))
            else:
                samplewise_iou.append(self.nanmean(area_inter[1:] / area_union[1:]))

            self.total_area_inter.scatter_(dim=0, index=class_dict, src=area_inter, reduce='add')
            self.total_area_union.scatter_(dim=0, index=class_dict, src=area_union, reduce='add')

            # above is equivalent to the following:
            '''
            self.total_area_inter[0] += area_inter[0].item()
            self.total_area_union[0] += area_union[0].item()
            for i in range(self.way + 1):
                self.total_area_inter[class_dict[i]] += area_inter[i].item()
                self.total_area_union[class_dict[i]] += area_union[i].item()
            '''

        if loss:
            bsz = float(pred_mask.shape[0])
            self.seg_loss_sum += loss * bsz
            self.seg_loss_count += bsz

        return torch.tensor(samplewise_iou) * 100.

    def nanmean(self, v):
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum() / (~is_nan).float().sum()

    def return_class_mapping_dict(self, support_classes):
        # [a, b] -> [0, a, b]
        # relative class index -> absolute class id
        bsz = support_classes.shape[0]
        bg_classes = torch.zeros(bsz, 1).to(support_classes.device).type(support_classes.dtype)
        class_dicts = torch.cat((bg_classes, support_classes), dim=1)
        return class_dicts

    def intersect_and_union(self, pred_mask, gt_mask):
        intersect = pred_mask[pred_mask == gt_mask]
        area_inter = torch.histc(intersect.float(), bins=(self.way + 1), min=0, max=self.way)
        area_pred_mask = torch.histc(pred_mask.float(), bins=(self.way + 1), min=0, max=self.way)
        area_gt_mask = torch.histc(gt_mask.float(), bins=(self.way + 1), min=0, max=self.way)
        area_union = area_pred_mask + area_gt_mask - area_inter
        return area_inter, area_union

    def compute(self):
        # miou does not include bg class
        inter_interest = self.total_area_inter[self.class_ids_interest]
        union_interest = self.total_area_union[self.class_ids_interest]
        iou_interest = inter_interest / torch.max(union_interest, self.ones)
        miou = torch.mean(iou_interest)

        '''
        fiou = inter_interest.sum() / union_interest.sum()
        biou = self.total_area_inter[0].sum() / self.total_area_union[0].sum()
        fbiou = (fiou + biou) / 2.
        '''
        return miou

    def compute_cls_er(self):
        return self.cls_er_sum / self.cls_er_count * 100. if self.cls_er_count else 0

    def avg_seg_loss(self):
        return self.seg_loss_sum / self.seg_loss_count if self.seg_loss_count else 0

    def avg_cls_loss(self):
        return self.cls_loss_sum / self.cls_loss_count if self.cls_loss_count else 0

    def update_cls(self, pred_cls, gt_cls, loss=None):
        pred_cls, gt_cls = pred_cls.cpu(), gt_cls.cpu()
        pred_correct = pred_cls == gt_cls
        bsz = float(pred_correct.shape[0])
        ''' accuracy '''
        # samplewise_acc = pred_correct.float().mean(dim=1)
        ''' exact ratio '''
        samplewise_er = torch.all(pred_correct, dim=1)
        self.cls_er_sum += samplewise_er.sum()
        self.cls_er_count += bsz

        if loss:
            self.cls_loss_sum += loss * bsz
            self.cls_loss_count += bsz

        return samplewise_er * 100.