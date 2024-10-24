import torch.nn.functional as F
import torch
import torch.nn as nn

from einops import rearrange
from .utils import get_reduction


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
        self,
        reduction: str = "mean",
        average: str = "macro",
        ignore_index=-100,
    ) -> None:
        super(DiceLoss, self).__init__()
        self.average = average
        self.ignore_index = ignore_index
        self.reduction = get_reduction(reduction)
        self.eps: float = 1e-6

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        class_weights: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
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

        # create the labels one hot tensort
        target_one_hot = target.clone()
        target_one_hot[target_one_hot == self.ignore_index] = input.shape[1]
        target_one_hot = F.one_hot(
            target_one_hot, num_classes=input.shape[1] + 1
        ).permute(0, 3, 1, 2)
        target_one_hot = target_one_hot[:, :-1, ::]

        if self.average == "macro":
            return self._macro_forward(
                input_soft, target_one_hot, class_weights=class_weights
            )
        # compute the actual dice score
        dice_score = self._calc_dice(input_soft, target_one_hot)
        return self.reduction(1.0 - dice_score)

    def _calc_dice(self, input, target):
        dims = (1, 2, 3)
        full_input = input.float() # Union could generate -Inf for fp16
        full_target = target.float()

        intersection = torch.sum(full_input * full_target, dims)
        cardinality = torch.sum(full_input + full_target, dims)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return dice_score

    def _macro_forward(self, input, target, class_weights=None):
        flat_input = rearrange(input, "b (c bin) h w -> (b c) bin h w", bin=1)
        flat_target = rearrange(target, "b (c bin) h w -> (b c) bin h w", bin=1)

        dice = 1.0 - self._calc_dice(flat_input, flat_target)  # (B X C)
        dice = rearrange(dice, "(b c) -> b c", c=input.shape[1])
        if class_weights is not None:
            dice = dice * class_weights
        dice = dice.mean(dim=1)
        return self.reduction(dice)
