import torch

from einops import rearrange

from label_anything.data.transforms import PromptsProcessor
from label_anything.data.utils import BatchKeys


def cartesian_product(a, b):
    # Create 1D tensors for indices along each dimension
    indices_a = torch.arange(a)
    indices_b = torch.arange(b)

    return torch.cartesian_prod(indices_a, indices_b)


def generate_points_from_errors(
    prediction: torch.tensor,
    ground_truth: torch.tensor,
    num_points: int,
    ignore_index: int = -100,
):
    """
    Generates a point for each class that can be positive or negative depending on the error being false positive or false negative.
    Args:
        prediction (torch.Tensor): The predicted segmentation mask of shape (batch_size, num_classes, height, width)
        ground_truth (torch.Tensor): The ground truth segmentation mask of shape (batch_size, num_classes, height, width)
        num_points (int): The number of points to generate for each class
    """
    B, C = prediction.shape[:2]
    device = prediction.device
    ground_truth = ground_truth.clone()
    ground_truth[ground_truth == ignore_index] = 0
    ground_truth = rearrange(
        torch.nn.functional.one_hot(ground_truth, C),
        "b h w c -> b c h w",
    )
    prediction = prediction.argmax(dim=1)
    prediction = rearrange(
        torch.nn.functional.one_hot(prediction, C),
        "b h w c -> b c h w",
    )
    errors = ground_truth - prediction
    coords = torch.nonzero(errors)
    if coords.shape[0] == 0:
        # No errors
        return (
            torch.zeros(B, C, 1, 2, device=device),
            torch.zeros(B, C, 1, device=device),
        )
    classes, counts = torch.unique(
        coords[:, 0:2], dim=0, return_counts=True, sorted=True
    )
    sampled_idxs = torch.cat(
        [torch.randint(0, x, (num_points,), device=device) for x in counts]
    ) + torch.cat([torch.tensor([0], device=device), counts.cumsum(dim=0)])[
        :-1
    ].repeat_interleave(
        num_points
    )
    sampled_points = coords[sampled_idxs]
    labels = errors[
        sampled_points[:, 0],
        sampled_points[:, 1],
        sampled_points[:, 2],
        sampled_points[:, 3],
    ]
    sampled_points = torch.index_select(
        sampled_points, 1, torch.tensor([0, 1, 3, 2], device=sampled_points.device)
    )  # Swap x and y
    all_classes = cartesian_product(B, C)
    missing = torch.tensor(
        list(
            set(tuple(elem) for elem in all_classes.tolist())
            - set(tuple(elem) for elem in classes.tolist())
        ),
        device=device,
    )
    missing = torch.cat([missing, torch.zeros(missing.shape, device=device)], dim=1)
    sampled_points = torch.cat([sampled_points, missing], dim=0)
    indices = (sampled_points[:, 0] * B + sampled_points[:, 1]).argsort()
    sampled_points = torch.index_select(sampled_points, 0, indices)

    labels = torch.cat([labels, torch.zeros(missing.shape[0], device=device)])
    labels = torch.index_select(labels, 0, indices)

    sampled_points = rearrange(
        sampled_points[:, 2:4],
        "(b c n) xy -> b c n xy",
        n=num_points,
        c=errors.shape[1],
    )
    labels = rearrange(labels, "(b c n) -> b c n", n=num_points, c=errors.shape[1])
    # ignore background
    labels[:, 0] = 0
    return sampled_points, labels


class Substitutor:
    """
    A class that cycle all the images in the examples as a query image.
    """

    torch_keys_to_exchange = [
        BatchKeys.PROMPT_POINTS,
        BatchKeys.PROMPT_MASKS,
        BatchKeys.PROMPT_BBOXES,
        BatchKeys.FLAG_MASKS,
        BatchKeys.FLAG_BBOXES,
        BatchKeys.FLAG_POINTS,
        BatchKeys.FLAG_EXAMPLES,
        BatchKeys.DIMS,
    ]
    torch_keys_to_separate = [
        BatchKeys.PROMPT_POINTS,
        BatchKeys.PROMPT_MASKS,
        BatchKeys.PROMPT_BBOXES,
        BatchKeys.FLAG_MASKS,
        BatchKeys.FLAG_BBOXES,
        BatchKeys.FLAG_POINTS,
        BatchKeys.FLAG_EXAMPLES,
    ]
    list_keys_to_exchange = [BatchKeys.INTENDED_CLASSES, BatchKeys.CLASSES, BatchKeys.IMAGE_IDS]
    list_keys_to_separate = []

    def __init__(
        self,
        threshold: float = None,
        num_points: int = 1,
        substitute=True,
        long_side_length=1024,
        custom_preprocess=True,
    ) -> None:
        self.example_classes = None
        self.threshold = threshold
        self.num_points = num_points
        self.substitute = self.calculate_if_substitute() and substitute
        self.it = 0
        self.prompt_processor = PromptsProcessor(long_side_length=long_side_length, custom_preprocess=custom_preprocess)

    def reset(self, batch: dict) -> None:
        self.it = 0
        self.batch, self.ground_truths = batch
        self.example_classes = self.batch[BatchKeys.CLASSES]

    def calculate_if_substitute(self):
        if self.threshold is None:
            return True
        return (
            torch.mean(
                torch.tensor(
                    [mean_pairwise_j_index(elem) for elem in self.example_classes]
                )
            )
            > self.threshold
        )

    def __iter__(self):
        return self

    def generate_new_points(self, prediction, ground_truth):
        """
        Generate new points from predictions errors and add them to the prompts
        """
        if self.substitute and self.num_points > 0:
            sampled_points, labels = generate_points_from_errors(
                prediction, ground_truth, self.num_points
            )
            sampled_points = torch.stack(
                [
                    self.prompt_processor.torch_apply_coords(elem, dim[0])
                    for dim, elem in zip(self.batch[BatchKeys.DIMS], sampled_points)
                ]
            )
            sampled_points = rearrange(sampled_points, "b c n xy -> b 1 c n xy")
            padding_points = torch.zeros(
                sampled_points.shape[0],
                self.batch[BatchKeys.PROMPT_POINTS].shape[1] - 1,
                *sampled_points.shape[2:],
                device=sampled_points.device,
            )
            labels = rearrange(labels, "b c n -> b 1 c n")
            padding_labels = torch.zeros(
                labels.shape[0],
                self.batch[BatchKeys.FLAG_POINTS].shape[1] - 1,
                *labels.shape[2:],
                device=labels.device,
            )
            sampled_points = torch.cat([sampled_points, padding_points], dim=1)
            labels = torch.cat([labels, padding_labels], dim=1)

            self.batch[BatchKeys.PROMPT_POINTS] = torch.cat(
                [self.batch[BatchKeys.PROMPT_POINTS], sampled_points], dim=3
            )
            self.batch[BatchKeys.FLAG_POINTS] = torch.cat(
                [self.batch[BatchKeys.FLAG_POINTS], labels], dim=3
            )

    def divide_query_examples(self):
        batch_examples = {}
        for key in self.torch_keys_to_separate:
            batch_examples[key] = self.batch[key][:, 1:]
        for key in self.list_keys_to_separate:
            batch_examples[key] = [elem[1:] for elem in self.batch[key]]
        gt = self.ground_truths[:, 0]
        for key in self.batch.keys() - set(
            self.torch_keys_to_separate + self.list_keys_to_separate
        ):
            batch_examples[key] = self.batch[key]

        return batch_examples, gt

    def __next__(self):
        torch_keys_to_exchange = self.torch_keys_to_exchange.copy()
        if "images" in self.batch:
            torch_keys_to_exchange.append("images")
            num_examples = self.batch["images"].shape[1]
            device = self.batch["images"].device
        if "embeddings" in self.batch:
            torch_keys_to_exchange.append("embeddings")
            if isinstance(self.batch["embeddings"], dict):
                # get the first key of the dict
                key = list(self.batch["embeddings"].keys())[0]
                num_examples = self.batch["embeddings"][key].shape[1]
                device = self.batch["embeddings"][key].device
            else:
                num_examples = self.batch["embeddings"].shape[1]
                device = self.batch["embeddings"].device

        if self.it == 0:
            self.it = 1
            return self.divide_query_examples()
        if not self.substitute:
            raise StopIteration
        if self.it == num_examples + 1:
            raise StopIteration
        if self.it == num_examples:  # Original query image becomes again query
            index_tensor = torch.cat(
                [
                    torch.tensor([num_examples - 1], device=device),
                    torch.arange(1, num_examples - 1, device=device),
                    torch.tensor([0], device=device),
                ]
            ).long()
        else:
            index_tensor = torch.cat(
                [
                    torch.tensor([self.it], device=device),
                    torch.arange(0, self.it, device=device),
                    torch.arange(self.it + 1, num_examples, device=device),
                ]
            ).long()

        for key in torch_keys_to_exchange:
            self.batch[key] = torch.index_select(
                self.batch[key], dim=1, index=index_tensor
            )

        for key in self.list_keys_to_exchange:
            if self.batch[key] is None:
                continue
            self.batch[key] = [
                [elem[i] for i in index_tensor] for elem in self.batch[key]
            ]
        for key in self.batch.keys() - set(
            self.torch_keys_to_exchange + self.list_keys_to_exchange
        ):
            self.batch[key] = self.batch[key]

        self.ground_truths = torch.index_select(
            self.ground_truths, dim=1, index=index_tensor
        )

        self.it += 1
        return self.divide_query_examples()
