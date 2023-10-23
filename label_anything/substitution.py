import torch


def cantor_function(x, y)
    return 0.5 * (x + y) * (x + y + 1) + y


def generate_points_from_errors(prediction, ground_truth, num_points):
    """
    Generates a point for each class that can be positive or negative depending on the error being false positive or false negative.
    Args:
        prediction (torch.Tensor): The predicted segmentation mask of shape (batch_size, num_classes, height, width)
        ground_truth (torch.Tensor): The ground truth segmentation mask of shape (batch_size, num_classes, height, width)
        num_points (int): The number of points to generate for each class
    """
    errors = ground_truth - prediction
    coords = torch.nonzero(errors)
    _, counts = torch.unique(coords[:, 0:2], dim=0, return_counts=True, sorted=True)
    sampled_idxs = torch.cat([torch.randint(0, x, (num_points,)) for x in counts]) + torch.cat([torch.tensor([0]), counts.cumsum(dim=0)])[:-1].repeat_interleave(num_points)
    sampled_points = coords[sampled_idx]
    labels = errors[sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], sampled_points[:, 3]]
    return sampled_points, labels