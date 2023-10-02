import logger
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Module, MSELoss
import torch.nn.functional as F
from einops import rearrange
from utils import instantiate_class, substitute_values
from ..models import FDOutput


def get_reduction(reduction: str):
    if reduction == 'none':
        return lambda x: x
    elif reduction == 'mean':
        return torch.mean
    elif reduction == 'sum':
        return torch.sum
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")


class CELoss(CrossEntropyLoss):
    def __init__(self, *args, weight=None, **kwargs):
        if weight:
            weight = torch.tensor(weight)
        super().__init__(*args, weight=weight, **kwargs)


class FocalLoss(Module):
    def __init__(self, gamma: float = 0, weight=None, reduction: str = 'mean', **kwargs):
        super().__init__()
        self.weight = None
        if weight:
            self.weight = torch.tensor(weight)
        self.gamma = gamma

        self.reduction = get_reduction(reduction)

    def __call__(self, x, target, **kwargs):
        ce_loss = F.cross_entropy(x, target, reduction='none', **kwargs)
        pt = torch.exp(-ce_loss)
        if self.weight is not None:
            self.weight = self.weight.to(x.device)
            wtarget = substitute_values(target, self.weight, unique=torch.arange(
                len(self.weight), device=target.device))
            focal_loss = torch.pow((1 - pt), self.gamma) * wtarget * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return self.reduction(focal_loss)


# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, reduction: str = 'mean', average: str = 'micro') -> None:
        super(DiceLoss, self).__init__()
        self.weight = None
        self.average = average
        if weight:
            self.weight = torch.tensor(weight)
            if self.average == 'micro':
                raise ValueError(
                    "Weighted Dice Loss is only supported for macro average")
        self.reduction = get_reduction(reduction)
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = F.one_hot(
            target, num_classes=input.shape[1]).permute(0, 3, 2, 1)

        if self.average == 'macro':
            return self._macro_forward(input_soft, target_one_hot)
        # compute the actual dice score
        dice_score = self._calc_dice(input_soft, target_one_hot)
        return self.reduction(1. - dice_score)

    def _calc_dice(self, input, target):
        dims = (1, 2, 3)

        intersection = torch.sum(input * target, dims)
        cardinality = torch.sum(input + target, dims)
        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)
        return dice_score

    def _macro_forward(self, input, target):
        flat_input = rearrange(input, 'b (c bin) h w -> (b c) bin h w', bin=1)
        flat_target = rearrange(
            target, 'b (c bin) h w -> (b c) bin h w', bin=1)

        dice = 1. - self._calc_dice(flat_input, flat_target)  # (B X C)
        dice = rearrange(dice, '(b c) -> b c', c=input.shape[1])
        if self.weight is not None:
            dice = dice * self.weight.to(input.device)
        dice = dice.mean(dim=1)
        return self.reduction(dice)


class VisklDivLoss(nn.KLDivLoss):
    """ KL divergence wrapper for Computer Vision tasks."""

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__(reduction='none')
        self.macro_reduction = get_reduction(reduction)

    def forward(self, student_output, teacher_output):
        return self.macro_reduction(
            super().forward(torch.log_softmax(student_output, dim=1),
                            torch.softmax(teacher_output, dim=1)).sum(dim=1)
        )


class ComposedLoss(nn.Module):
    name = "ComposedLoss"

    def __init__(self) -> None:
        super().__init__()
        self.__class__.__name__ = self.name

    def component_names(self):
        raise NotImplementedError(
            "Component names not implemented for ComposedLoss abstract class")


class AuxiliaryLoss(ComposedLoss):
    name = "AuxLoss"
    """ Auxiliary loss, wraps the task loss and auxiliary loss """

    def __init__(self, task_loss_fn, aux_loss_fn, aux_loss_coeff: float = 0.2, **kwargs):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.aux_loss_coeff = aux_loss_coeff

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return [self.name,
                self.task_loss_fn.__class__.__name__,
                self.aux_loss_fn.__class__.__name__
                ]

    def forward(self, task_aux_output, target):
        out, aux = task_aux_output
        task_loss = self.task_loss_fn(out, target)
        if isinstance(task_loss, tuple):  # SOME LOSS FUNCTIONS RETURNS LOSS AND LOG_ITEMS
            task_loss = task_loss[0]
        aux_loss = self.aux_loss_fn(aux['aux'], target)
        loss = task_loss * (1 - self.aux_loss_coeff) + \
            aux_loss * self.aux_loss_coeff

        return loss, torch.cat((loss.unsqueeze(0), task_loss.unsqueeze(0), aux_loss.unsqueeze(0))).detach()


class KDLogitsLoss(ComposedLoss):
    name = "KDLLoss"
    """ Knowledge distillation loss, wraps the task loss and distillation loss """

    def __init__(self, task_loss_fn, distillation_loss_fn=VisklDivLoss(), distillation_loss_coeff: float = 0.5):
        '''
        :param task_loss_fn: task loss. E.g., LabelSmoothingCrossEntropyLoss
        :param distillation_loss_fn: distillation loss. E.g., KLDivLoss
        :param distillation_loss_coeff:
        '''

        super(KDLogitsLoss, self).__init__()
        self.task_loss_fn = task_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.distillation_loss_coeff = distillation_loss_coeff

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return [self.name,
                self.task_loss_fn.__class__.__name__,
                self.distillation_loss_fn.__class__.__name__
                ]

    def forward(self, kd_module_output, target):
        task_loss = self.task_loss_fn(kd_module_output.student_output, target)
        if isinstance(task_loss, tuple):  # SOME LOSS FUNCTIONS RETURNS LOSS AND LOG_ITEMS
            task_loss = task_loss[0]
        distillation_loss = self.distillation_loss_fn(
            kd_module_output.student_output, kd_module_output.teacher_output)
        loss = task_loss * (1 - self.distillation_loss_coeff) + \
            distillation_loss * self.distillation_loss_coeff

        return loss, torch.cat((loss.unsqueeze(0), task_loss.unsqueeze(0), distillation_loss.unsqueeze(0))).detach()


class KDFeatureLogitsLoss(ComposedLoss):
    name = "KDFLLoss"

    def __init__(self, task_loss_fn, feature_loss_fn=MSELoss(), distillation_loss_fn=VisklDivLoss(),
                 dist_feats_loss_coeffs=(0.2, 0.4, 0.4), feats_loss_reduction='mean', **kwargs):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.feature_loss_fn = feature_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.dist_feats_loss_coeff = dist_feats_loss_coeffs
        self.feats_loss_reduction = get_reduction(feats_loss_reduction)

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return [self.name,
                self.task_loss_fn.__class__.__name__,
                self.distillation_loss_fn.__class__.__name__,
                self.feature_loss_fn.__class__.__name__
                ]

    def forward(self, kd_output: FDOutput, target):
        logits_loss = self.distillation_loss_fn(
            kd_output.student_output, kd_output.teacher_output)
        feats_loss = self.feats_loss_reduction(torch.tensor([
            self.feature_loss_fn(student_feat, teacher_feat)
            for student_feat, teacher_feat in zip(kd_output.student_features, kd_output.teacher_features
                                                  )], device=kd_output.student_output.device))
        task_loss = self.task_loss_fn(kd_output.student_output, target)

        loss = task_loss * self.dist_feats_loss_coeff[0] + \
            logits_loss * self.dist_feats_loss_coeff[1] + \
            feats_loss * self.dist_feats_loss_coeff[2]

        return loss, torch.cat((loss.unsqueeze(0),
                                task_loss.unsqueeze(0),
                                logits_loss.unsqueeze(0),
                                feats_loss.unsqueeze(0)
                                )).detach()


class VariationalInformationLoss(nn.Module):
    def forward(self, student_feats, teacher_feats, **kwargs):
        mu, sigma = student_feats
        return torch.sum(
            torch.log(sigma) + torch.div(torch.square(teacher_feats - mu),
                                         2 * torch.square(sigma))
        )


class VariationalInformationLossMean(nn.Module):
    def forward(self, student_feats, teacher_feats, **kwargs):
        mu, sigma = student_feats
        return torch.mean(
            torch.log(sigma) + torch.div(torch.square(teacher_feats - mu),
                                         2 * torch.square(sigma))
        )


class VariationalInformationLossScaled(VariationalInformationLoss):
    def __init__(self, scale_factor=1e-6, *args, **krwargs) -> None:
        super().__init__(*args, **krwargs)
        self.factor = scale_factor

    def forward(self, student_feats, teacher_feats, **kwargs):
        return super().forward(student_feats, teacher_feats, **kwargs) * self.factor


class VariationalInformationLogitsLoss(KDFeatureLogitsLoss):
    name = "VILLoss"
    variants = {
        'standard': VariationalInformationLoss,
        'mean': VariationalInformationLossMean,
        'scaled': VariationalInformationLossScaled,
    }

    def __init__(self, task_loss_fn, distillation_loss_fn=VisklDivLoss(),
                 dist_feats_loss_coeffs=(0.2, 0.4, 0.4), feats_loss_reduction='mean',
                 variant='standard', scale_factor=1e-6):
        params = {'scale_factor': scale_factor} if variant == 'scaled' else {}
        feature_loss_fn = self.variants[variant](**params)
        super().__init__(task_loss_fn=task_loss_fn,
                         feature_loss_fn=feature_loss_fn,
                         distillation_loss_fn=distillation_loss_fn,
                         dist_feats_loss_coeffs=dist_feats_loss_coeffs,
                         feats_loss_reduction=feats_loss_reduction)

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        comps = super().component_names
        comps[0] = self.name
        return comps


LOSSES = {
    'cross_entropy': CELoss,
    'dice': DiceLoss,
    'focal': FocalLoss,
    'variational_information_loss': VariationalInformationLoss,
    'mean_variational_information_loss': VariationalInformationLossMean,
    'scaled_variational_information_loss': VariationalInformationLossScaled,
    "vis_kldiv_loss": VisklDivLoss,

    # KD composed losses
    'kd_logits_loss': KDLogitsLoss,
    'kd_feature_logits_loss': KDFeatureLogitsLoss,
    'variational_information_logits_loss': VariationalInformationLogitsLoss,

    # Auxiliary losses
    "auxiliary_loss": AuxiliaryLoss
}


def instiantiate_loss(loss_name, params):
    """
    Instantiate a loss function from a string name.
    Args:
        loss_name (str): Name of the loss function.
        params (dict): Parameters for the loss function.
    Returns:
        loss_fn (nn.Module): Loss function.
    """
    try:
        return instantiate_class(loss_name, params)
    except (AttributeError, ValueError) as ex:
        logger.info(f'Loss {loss_name} not instantiable from local module.')

        loss_cls_names = {
            loss_cls.__name__: loss_cls for loss_cls in LOSSES.values()}
        if loss_name in LOSSES:
            return LOSSES[loss_name](**params)
        elif loss_name in loss_cls_names:
            return loss_cls_names[loss_name](**params)
        elif loss_name in nn.__dict__:
            return nn.__dict__[loss_name](**params)
        else:
            raise ValueError(
                f'Loss {loss_name} not found. Available losses: {list(LOSSES.keys())}'
            ) from ex
