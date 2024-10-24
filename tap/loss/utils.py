import torch

from tap.utils.utils import substitute_values


def get_reduction(reduction: str):
    if reduction == "none":
        return lambda x: x
    elif reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    
    
def get_weight_matrix_from_labels(labels, num_classes, ignore_index=-100):  
    there_is_ignore = ignore_index in labels
    if there_is_ignore:
        weight_labels = labels.clone()
        weight_labels += 1
        weight_labels[weight_labels == ignore_index + 1] = 0
        weight_num_classes = num_classes + 1
    else:
        weight_labels = labels
        weight_num_classes = num_classes
    weights = torch.ones(weight_num_classes, device=labels.device)
    classes, counts = weight_labels.unique(return_counts=True)
    classes = classes.long()
    if there_is_ignore:
        weights[classes] = 1 / torch.log(1.1 + counts / counts.sum())
        weights[0] = 0
        class_weights = weights[1:]
    else:
        weights[classes] = 1 / torch.log(1.1 + counts / counts.sum())
        class_weights = weights
    wtarget = substitute_values(
        weight_labels,
        weights,
        unique=torch.arange(weight_num_classes, device=labels.device),
    )
    return wtarget, class_weights