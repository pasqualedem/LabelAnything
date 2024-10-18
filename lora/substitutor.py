import numpy as np
import torch

from einops import rearrange

from label_anything.data.transforms import PromptsProcessor
from label_anything.data.utils import BatchKeys


def divide_query_examples(
    batch, ground_truths, torch_keys_to_separate, list_keys_to_separate
):
    batch_examples = {key: batch[key][:, 1:] for key in torch_keys_to_separate}
    for key in list_keys_to_separate:
        batch_examples[key] = [elem[1:] for elem in batch[key]]
    gt = ground_truths[:, 0]
    for key in batch.keys() - set(torch_keys_to_separate + list_keys_to_separate):
        batch_examples[key] = batch[key]

    return batch_examples, gt


def generate_incremental_tensors(N, K):
    example_matrix = torch.arange(N * K).reshape(N, K).T
    ordered_elements = example_matrix.flatten().tolist()
    result_tensors = []

    for i, num in enumerate(ordered_elements):
        # Start the tensor with the current element i (starting from 0)
        current_tensor = torch.tensor([num], dtype=torch.int32)

        # Determine how many extra elements to include based on iteration
        # Each tensor has 1 + (extra elements). The number of extra elements grows by N every N iterations
        num_extra_rows = (
            (i // N) + 1
        )  # Extra elements = N for first batch, 2N for second, etc.

        i_row = i // N  # The row of the current element in the example matrix
        if i_row == K - 1:
            # If the current element is the last element in the example matrix, skip
            # because there are no more rows to sample from
            extra_elements = torch.tensor([x for x in ordered_elements if x != num])
        else:
            possible_rows = [row_k for row_k in range(K) if row_k != i_row]
            sampled_rows = np.random.choice(possible_rows, num_extra_rows, replace=False)
            sampled_rows = torch.tensor(np.sort(sampled_rows))
            rows = torch.index_select(example_matrix, 0, sampled_rows)
            extra_elements = rearrange(rows, "b n -> (b n)", n=N)

        # Combine the starting element with the extra elements
        current_tensor = torch.cat((current_tensor, extra_elements))

        # Add the tensor to the result list
        result_tensors.append(current_tensor)

    return result_tensors


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
        self, substitute=True, long_side_length=1024, custom_preprocess=True, **kwargs
    ) -> None:
        if kwargs:
            print(f"Warning: Unrecognized arguments: {kwargs}")
        self.example_classes = None
        self.substitute = substitute
        self.it = 0
        self.query_iteration = True
        self.prompt_processor = PromptsProcessor(
            long_side_length=long_side_length, custom_preprocess=custom_preprocess
        )

    def reset(self, batch: dict) -> None:
        self.it = 0
        self.batch, self.ground_truths = batch
        self.batch, self.query_image_gt_dim = self.first_divide_query_examples()
        self.example_classes = self.batch[BatchKeys.CLASSES]
        self.query_iteration = True

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

    def divide_query_examples(self, batch=None, gt=None):
        batch = self.batch if batch is None else batch
        gt = self.ground_truths if gt is None else gt
        return divide_query_examples(
            batch, gt, self.torch_keys_to_separate, self.list_keys_to_separate
        )

    def divide_query_examples_append_query_dim(self):
        batch, gt = self.divide_query_examples()
        batch = {
            **batch,
            "dims": torch.cat(
                [batch["dims"], self.query_image_gt_dim[2].unsqueeze(1)], dim=1
            ),
        }
        return batch, gt

    def get_batch_info(self):
        torch_keys_to_exchange = self.torch_keys_to_exchange.copy()
        if "images" in self.batch:
            torch_keys_to_exchange.append("images")
            num_images = self.batch["images"].shape[1]
            device = self.batch["images"].device
        if "embeddings" in self.batch:
            torch_keys_to_exchange.append("embeddings")
            num_images = self.batch["embeddings"].shape[1]
            device = self.batch["embeddings"].device
        return torch_keys_to_exchange, num_images, device

    def _query_iteration(self):
        self.query_iteration = False
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

    def __next__(self):
        torch_keys_to_exchange, num_images, device = self.get_batch_info()

        if self.query_iteration:
            return self._query_iteration()
        if self.it == 0:
            self.it = 1
            return self.divide_query_examples_append_query_dim()

        if not self.substitute:
            raise StopIteration
        if self.it == num_images:
            raise StopIteration
        else:
            index_tensor = torch.cat(
                [
                    torch.tensor([self.it], device=device),
                    torch.arange(0, self.it, device=device),
                    torch.arange(self.it + 1, num_images, device=device),
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
            torch_keys_to_exchange + self.list_keys_to_exchange
        ):
            self.batch[key] = self.batch[key]

        self.ground_truths = torch.index_select(
            self.ground_truths, dim=1, index=index_tensor
        )

        self.it += 1
        return self.divide_query_examples_append_query_dim()


class IncrementalSubstitutor(Substitutor):
    def __init__(
        self,
        substitute=True,
        long_side_length=1024,
        custom_preprocess=True,
        n_ways=None,
        k_shots=None,
    ):
        super().__init__(substitute, long_side_length, custom_preprocess)
        self.n_ways = n_ways
        self.k_shots = k_shots
        super().__init__(substitute, long_side_length, custom_preprocess)
        self.index_tensors = generate_incremental_tensors(self.n_ways, self.k_shots)
        
    def reset(self, batch):
        self.index_tensors = generate_incremental_tensors(self.n_ways, self.k_shots)
        return super().reset(batch)

    def divide_query_examples_append_dims(self, batch, gt):
        batch = {
            **batch,
            "dims": torch.cat(
                [batch["dims"], self.query_image_gt_dim[2].unsqueeze(1)], dim=1
            ),
        }
        return self.divide_query_examples(batch, gt)

    def __next__(self):
        torch_keys_to_exchange, num_images, device = self.get_batch_info()
        torch_keys_to_exchange.remove(BatchKeys.DIMS)
        if self.query_iteration:
            return self._query_iteration()
        if not self.substitute:
            raise StopIteration
        if self.it == num_images:
            raise StopIteration
        else:
            index_tensor = self.index_tensors[self.it].to(device)

        new_batch = {
            key: torch.index_select(self.batch[key], dim=1, index=index_tensor)
            for key in torch_keys_to_exchange
        }
        for key in self.list_keys_to_exchange:
            new_batch[key] = [
                [elem[i] for i in index_tensor] for elem in self.batch[key]
            ]
        # Add the remaning DIMS
        dims_index = torch.cat(
            (
                index_tensor,
                torch.tensor(
                    [x for x in range(num_images) if x not in index_tensor],
                    device=device,
                    dtype=torch.long,
                ),
            )
        )
        new_batch[BatchKeys.DIMS] = torch.index_select(
            self.batch[BatchKeys.DIMS], dim=1, index=dims_index
        )

        for key in self.batch.keys() - set(
            torch_keys_to_exchange + self.list_keys_to_exchange
        ):
            new_batch[key] = self.batch[key]

        new_gt = torch.index_select(self.ground_truths, dim=1, index=index_tensor)

        self.it += 1
        return self.divide_query_examples_append_dims(new_batch, new_gt)
