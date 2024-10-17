import torch

from einops import rearrange

from label_anything.data.transforms import PromptsProcessor
from label_anything.data.utils import BatchKeys


def cartesian_product(a, b):
    # Create 1D tensors for indices along each dimension
    indices_a = torch.arange(a)
    indices_b = torch.arange(b)

    return torch.cartesian_prod(indices_a, indices_b)


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
    list_keys_to_exchange = [BatchKeys.CLASSES, BatchKeys.IMAGE_IDS]
    list_keys_to_separate = []

    def __init__(
        self,
        substitute=True,
        long_side_length=1024,
        custom_preprocess=True,
    ) -> None:
        self.example_classes = None
        self.substitute = substitute
        self.it = 0
        self.prompt_processor = PromptsProcessor(long_side_length=long_side_length, custom_preprocess=custom_preprocess)

    def reset(self, batch: dict) -> None:
        self.it = 0
        self.batch, self.ground_truths = batch
        self.batch, self.query_image_gt_dim = self.first_divide_query_examples()
        self.example_classes = self.batch[BatchKeys.CLASSES]

    def __iter__(self):
        return self
    
    def first_divide_query_examples(self):
        batch_examples, query_gt = self.divide_query_examples()
        self.ground_truths = self.ground_truths[:, 1:]
        query_image = batch_examples["images"][:, 0]
        batch_examples["images"] = batch_examples["images"][:, 1:]
        query_dims = batch_examples["dims"][:, 0]
        batch_examples["dims"] = batch_examples["dims"][:, 1:]
        return batch_examples, (query_image, query_gt, query_dims)

    def divide_query_examples(self):
        batch_examples = {
            key: self.batch[key][:, 1:] for key in self.torch_keys_to_separate
        }
        for key in self.list_keys_to_separate:
            batch_examples[key] = [elem[1:] for elem in self.batch[key]]
        gt = self.ground_truths[:, 0]
        for key in self.batch.keys() - set(
            self.torch_keys_to_separate + self.list_keys_to_separate
        ):
            batch_examples[key] = self.batch[key]

        return batch_examples, gt
    
    def divide_query_examples_append_query_dim(self):
        batch, gt = self.divide_query_examples()
        batch = {
            **batch,
            "dims": torch.cat([batch["dims"], self.query_image_gt_dim[2].unsqueeze(1)], dim=1),
        }
        return batch, gt

    def __next__(self):
        torch_keys_to_exchange = self.torch_keys_to_exchange.copy()
        if "images" in self.batch:
            torch_keys_to_exchange.append("images")
            num_examples = self.batch["images"].shape[1]
            device = self.batch["images"].device
        if "embeddings" in self.batch:
            torch_keys_to_exchange.append("embeddings")
            num_examples = self.batch["embeddings"].shape[1]
            device = self.batch["embeddings"].device

        if self.it == 0:
            self.it = 1
            batch = {
                **self.batch,
                "images": torch.cat(
                    [self.query_image_gt_dim[0].unsqueeze(1), self.batch["images"]], dim=1
                ),
                "dims": torch.cat(
                    [self.query_image_gt_dim[2].unsqueeze(1), self.batch["dims"]], dim=1
                ),
            }
            return batch, self.query_image_gt_dim[1]
        if self.it == 1:
            self.it = 2
            return self.divide_query_examples_append_query_dim()
            
        if not self.substitute:
            raise StopIteration
        if self.it == num_examples:
            raise StopIteration
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
        return self.divide_query_examples_append_query_dim()
