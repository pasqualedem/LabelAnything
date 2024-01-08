import torch


def get_reduction(reduction: str):
    if reduction == "none":
        return lambda x: x
    elif reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")