import torch


def iou(pred: torch.Tensor, target: torch.Tensor):
    """Compute the intersection over union (IoU) per class.

    Args:
        pred (torch.Tensor): A tensor of shape (N, C, H, W) representing the predicted masks.
        target (torch.Tensor): A tensor of shape (N, H, W) representing the true masks.

    Returns:
        ious (list): A list of length C containing the IoU for each class.
    """
    ious = []
    num_classes = pred.shape[1]
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = (
            pred_inds.long().sum().item()
            + target_inds.long().sum().item()
            - intersection
        )
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious
