import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Module, MSELoss
import torch.nn.functional as F
from einops import rearrange
from .utils import instantiate_class, substitute_values


def get_reduction(reduction: str):
    if reduction == "none":
        return lambda x: x
    elif reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")


class FocalLoss(Module):
    def __init__(
        self, gamma: float = 0, weight=None, reduction: str = "mean", **kwargs
    ):
        super().__init__()
        self.weight = None
        if weight:
            self.weight = torch.tensor(weight)
        self.gamma = gamma

        self.reduction = get_reduction(reduction)

    def __call__(self, x, target, **kwargs):
        ce_loss = F.cross_entropy(x, target, reduction="none", **kwargs)
        pt = torch.exp(-ce_loss)
        if self.weight is not None:
            self.weight = self.weight.to(x.device)
            wtarget = substitute_values(
                target,
                self.weight,
                unique=torch.arange(len(self.weight), device=target.device),
            )
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

    def __init__(
        self, weight=None, reduction: str = "mean", average: str = "micro"
    ) -> None:
        super(DiceLoss, self).__init__()
        self.weight = None
        self.average = average
        if weight:
            self.weight = torch.tensor(weight)
            if self.average == "micro":
                raise ValueError(
                    "Weighted Dice Loss is only supported for macro average"
                )
        self.reduction = get_reduction(reduction)
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(input))
            )
        if not len(input.shape) == 4:
            raise ValueError(
                "Invalid input shape, we expect BxNxHxW. Got: {}".format(input.shape)
            )
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError(
                "input and target shapes must be the same. Got: {}".format(
                    input.shape, input.shape
                )
            )
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device
                )
            )
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(
            0, 3, 2, 1
        )

        if self.average == "macro":
            return self._macro_forward(input_soft, target_one_hot)
        # compute the actual dice score
        dice_score = self._calc_dice(input_soft, target_one_hot)
        return self.reduction(1.0 - dice_score)

    def _calc_dice(self, input, target):
        dims = (1, 2, 3)

        intersection = torch.sum(input * target, dims)
        cardinality = torch.sum(input + target, dims)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return dice_score

    def _macro_forward(self, input, target):
        flat_input = rearrange(input, "b (c bin) h w -> (b c) bin h w", bin=1)
        flat_target = rearrange(target, "b (c bin) h w -> (b c) bin h w", bin=1)

        dice = 1.0 - self._calc_dice(flat_input, flat_target)  # (B X C)
        dice = rearrange(dice, "(b c) -> b c", c=input.shape[1])
        if self.weight is not None:
            dice = dice * self.weight.to(input.device)
        dice = dice.mean(dim=1)
        return self.reduction(dice)
    

class LabelAnythingLoss(nn.Module):
    """This loss is a linear combination of the following losses:
    - FocalLoss
    - DiceLoss
    """
    def __init__(self, focal_weight=20, dice_weight=1):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=2)
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return (
            self.focal_weight * self.focal_loss(logits, target)
            + self.dice_weight * self.dice_loss(logits, target)
        )


if __name__ == "__main__":
    criterion = LabelAnythingLoss()
    logits = torch.randn(
        2, 3, 256, 256
    )  # batch size, number of classes, height, width
    target = torch.randint(0, 3, (2, 256, 256))
    loss = criterion(logits, target)
    print(loss)  

