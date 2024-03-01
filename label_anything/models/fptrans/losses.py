import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt


def get(opt, logger, loss=None):
    loss = loss or opt.loss
    if loss == "ce":
        logger.info(' ' * 5 + "==> CrossEntropyLoss is used.")
        return nn.CrossEntropyLoss(ignore_index=255)
    elif loss == "cedt":
        logger.info(' ' * 5 + "==> CELossWithDT is used.")
        return CELossWithDT(opt.sigma, opt.precompute_weight)
    elif loss == 'pairwise':
        logger.info(' ' * 5 + "==> IntraImageContrastLoss is used.")
        return PairwiseLoss()
    else:
        raise ValueError(f"Unsupported loss type, got {opt.loss}. Please choose from [ce, cedt, pairwise]")


class CELossWithDT(object):
    def __init__(self, sigma, precompute_weight=False):
        self.sigma = sigma
        self.loss_obj = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        if not precompute_weight:
            self.kernel = torch.ones(1, 1, 3, 3, dtype=torch.float).cuda()

    def boundary2weight(self, boundary):
        bool_boundary = np.around(boundary.detach().cpu().numpy()).astype(np.bool)
        edts = []
        for bdn in bool_boundary:
            edt = distance_transform_edt(np.bitwise_not(bdn))
            edts.append(edt)
        edts_t = torch.from_numpy(np.stack(edts, axis=0))
        weight = torch.exp(-edts_t / self.sigma ** 2) + 1
        return weight.to(dtype=torch.float32, device=boundary.device)

    def __call__(self, inputs, target, weight=None):
        """
        inputs: [b, c, h, w]
        target: [b, h, w]
        """
        loss = self.loss_obj(inputs, target)
        if weight is None:
            mask = torch.zeros_like(target, dtype=torch.float32)
            mask[target == 1] = 1
            mask.unsqueeze_(dim=1)      # [bs, 1, H, W]
            dilated = torch.clamp(F.conv2d(mask, self.kernel, padding=1), 0, 1) - mask
            erosion = mask - torch.clamp(F.conv2d(mask, self.kernel, padding=1) - 8, 0, 1)
            boundary = (dilated + erosion).squeeze(dim=1)     # [bs, H, W]
            weight = self.boundary2weight(boundary)
        else:
            weight = weight + 1
        weighted_loss = loss * weight
        return weighted_loss.sum() / weight.sum()


class PairwiseLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.scale = 10
        self.reduction = reduction
        self.loss_obj = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, x1, y1, x2, y2):
        """
        x1: torch.Tensor. [B, S, C, N]
        x2: torch.Tensor. [B, 1, C, N]
        y1: torch.Tensor. [B, S, N], containing {0, 1, 255}
        y2: torch.Tensor. [B, 1, N], containing {0, 1, 255}
        """
        B, S, C, N = x1.shape
        
        x1 = F.normalize(x1, p=2, dim=2)    # [B, S, C, N]
        x1 = x1.permute(0, 1, 3, 2).reshape(B, S * N, C)        # [B, N1, C]
        y1 = y1.view(B, S * N, 1)                               # [B, N1, 1]
        x2 = F.normalize(x2, p=2, dim=2)    # [BQ, C, N]
        x2 = x2.permute(0, 2, 1, 3).reshape(B, C, N)            # [B, C, N2]
        y2 = y2.view(B, 1, N)                                   # [B, 1, N2]

        sim = torch.bmm(x1, x2)                                 # [B, N1, N2]
        lab = (y1 == y2).float()                                # [B, N1, N2]
        ignore = y1 + y2 >= 255
        keep = ~(ignore | (y1 + y2 == 0))

        loss_no_reduce = self.loss_obj(sim * self.scale, lab)
        if self.reduction == 'none':
            return loss_no_reduce, keep
        elif self.reduction == 'mean':
            # Compute on used positions
            loss = (loss_no_reduce * keep).sum() / (keep.sum() + 1e-6)
            return loss
        else:
            raise ValueError
