import torch

from einops import rearrange

from data.utils import mean_pairwise_j_index


def generate_points_from_errors(
    prediction: torch.tensor, ground_truth: torch.tensor, num_points: int
):
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
    sampled_idxs = torch.cat(
        [torch.randint(0, x, (num_points,)) for x in counts]
    ) + torch.cat([torch.tensor([0]), counts.cumsum(dim=0)])[:-1].repeat_interleave(
        num_points
    )
    sampled_points = coords[sampled_idxs]
    labels = errors[
        sampled_points[:, 0],
        sampled_points[:, 1],
        sampled_points[:, 2],
        sampled_points[:, 3],
    ]
    sampled_points = rearrange(
        sampled_points[:, 2:4],
        "(b c n) xy -> b c n xy",
        n=num_points,
        c=errors.shape[1],
    )
    labels = rearrange(labels, "(b c n) -> b c n", n=num_points, c=errors.shape[1])
    return sampled_points, labels


class Substitutor:
    """
    A class that cycle all the images in the examples as a query image.
    """

    def __init__(self, batch: dict, threshold: float, num_points: int) -> None:
        self.batch = batch
        self.example_classes = batch["example_classes"]
        self.threshold = threshold
        self.num_points = num_points
        self.substitute = self.calculate_if_substitute()
        self.it = 0

    def calculate_if_substitute(self):
        return mean_pairwise_j_index(self.example_classes) > self.threshold

    def __iter__(self):
        return self

    def generate_new_points(self, prediction, ground_truth):
        """
        Generate new points from predictions errors and add them to the prompts
        """
        sampled_points, labels = generate_points_from_errors(
            prediction, ground_truth, self.num_points
        )
        sampled_points = rearrange(sampled_points, "b c n xy -> b 1 c n xy")
        padding_points = torch.zeros(
            sampled_points.shape[0],
            self.batch["prompt_points"].shape[1] - 1,
            *sampled_points.shape[2:],
        )
        labels = rearrange(labels, "b c n -> b 1 c n")
        padding_labels = torch.zeros(
            labels.shape[0],
            self.batch["prompt_point_labels"].shape[1] - 1,
            *labels.shape[2:],
        )
        sampled_points = torch.cat([padding_points, sampled_points], dim=1)
        labels = torch.cat([padding_labels, labels], dim=1)

        self.batch["prompt_points"] = torch.cat(
            [self.batch["prompt_points"], sampled_points], dim=3
        )
        self.batch["prompt_points_labels"] = torch.cat(
            [self.batch["prompt_point_labels"], labels], dim=3
        )

    def __next__(self):
        if self.it == 0:
            self.it = 1
            return self.batch
        if not self.substitute:
            raise StopIteration
        num_examples = self.batch["example_classes"].shape[0]
        if self.it == num_examples:
            raise StopIteration

        index_tensor = torch.cat(
            [
                torch.tensor([self.it]),
                torch.arange(0, self.it),
                torch.arange(self.it + 1, num_examples),
            ]
        ).long()

        keys_to_exchange = [
            "prompt_points",
            "prompt_point_labels",
            "prompt_masks",
            "prompt_boxes",
            "flags_masks",
            "flags_boxes",
            "flags_points",
            "ground_truth",
        ]

        if "images" in self.batch:
            keys_to_exchange.append("images")
        elif "embeddings" in self.batch:
            keys_to_exchange.append("embeddings")
        else:
            raise ValueError("Batch must contain either images or embeddings")

        for key in keys_to_exchange:
            self.batch[key] = torch.index_select(
                self.batch[key], dim=1, index=index_tensor
            )

        self.it += 1
        return self.batch
