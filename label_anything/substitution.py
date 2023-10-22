import torch


def generate_points_from_errors(prediction, ground_truth, num_points):
    """
    Generates a point for each class that can be positive or negative depending on the error being false positive or false negative.
    Args:
        prediction (torch.Tensor): The predicted segmentation mask of shape (batch_size, num_classes, height, width)
        ground_truth (torch.Tensor): The ground truth segmentation mask of shape (batch_size, num_classes, height, width)
        num_points (int): The number of points to generate for each class
    """
    errors = ground_truth - prediction
    coords = [torch.nonzero(errors[i]) for i in range(errors.shape[0])]
    sampled_idxs = [torch.randint(0, coord_dim.shape[0], (num_points,)) for coord_dim in coords]
    sampled_points = torch.cat([coords[dim][idx] for dim, idx in enumerate(sampled_idxs)])
    sampled_points = torch.cat([torch.arange(errors.shape[0]).repeat_interleave(num_points).unsqueeze(0).T, sampled_points], dim=1)
    labels = errors[sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]]
    return sampled_points, labels