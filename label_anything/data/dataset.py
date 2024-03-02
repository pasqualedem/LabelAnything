import random
import torch
import itertools

from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset, BatchSampler

import label_anything.data.utils as utils
from label_anything.data.coco import CocoLVISDataset
from label_anything.data.coco20i import Coco20iDataset
from label_anything.logger.text_logger import get_logger

logger = get_logger(__name__)


datasets = {
    "coco": CocoLVISDataset,
    "val_coco": CocoLVISDataset,
    "coco20i": Coco20iDataset,
    "val_coco20i": Coco20iDataset,
    "lvis": CocoLVISDataset,
    # "voc": PascalVOCDataset,
    # "val_voc": PascalVOCDataset,
    "voc5i": Coco20iDataset,
    "val_voc5i": Coco20iDataset,
    "val_lvis": CocoLVISDataset,
}


class LabelAnythingDataset(Dataset):
    def __init__(self, datasets_params: Dict, common_params: Dict) -> None:
        """
        Initializes a LabelAnythingDataset Dataset object.

        Args:
            datasets_params (Dict): A dictionary containing the parameters for each dataset.
            common_params (Dict): A dictionary containing the common parameters for all datasets.
        """

        self.datasets = {
            dataset_name: datasets[dataset_name](**{**common_params, **params})
            for dataset_name, params in datasets_params.items()
        }
        self.categories = {
            dataset_name: dataset.categories
            for dataset_name, dataset in self.datasets.items()
        }
        index = sum(
            [
                [(dataset_name, i) for i in range(len(dataset))]
                for dataset_name, dataset in self.datasets.items()
            ],
            [],
        )
        self.index = {i: index for i, index in enumerate(index)}

        super().__init__()

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets.values()])

    def __getitem__(self, idx_metadata) -> Any:
        """
        Returns the item at the given index.

        Args:
            idx_num_examples: A tuple containing the index and the number of examples.

        Returns:
            Any: The item at the given index.
        """
        idx, batch_metadata = idx_metadata
        dataset_name, dataset_index = self.index[idx]
        return (
            self.datasets[dataset_name][(dataset_index, batch_metadata)],
            dataset_name,
        )

    def load_and_preprocess_images(self, dataset_name, image_ids):
        return self.datasets[dataset_name].load_and_preprocess_images(image_ids)

    def collate_fn(
        self, batched_input: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Performs the collate_fn, which is useful for batching data points in a dataloader.

        Args:
            batched_input (List[Dict[str, Any]]): A list of batch_size elements, where each element is a dictionary
                containing the following entries:
                - 'target': query image as a torch tensor of shape 3 x H x W.
                - 'examples': example image as a torch tensor of shape M x 3 x H x W, where M is the number of examples
                    extracted for the given query image.
                - 'prompt_mask': example image masks as a torch tensor of shape M x C x H x W, where M is the number of
                    examples extracted for the given query image and C is the number of classes associated with it.
                - 'prompt_coords': example image coordinates as a torch tensor of shape M x C x N x K x 2, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated with the
                    given image, N is the maximum number of annotations associated with a pair (image, class), and K is
                    the number of points extracted.
                - 'flag_coords': example image coordinate flags as a torch tensor of shape M x C x N x K, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated with the
                    given image, N is the maximum number of annotations associated with a pair (image, class), and K is
                    the number of points extracted.
                - 'prompt_bbox': example image bounding boxes as a torch tensor of shape M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated with the
                    given image, and N is the maximum number of annotations associated with a pair (image, class). The
                    last dimension is 4 because a single bounding box is represented by the top-left and bottom-right
                    coordinates.
                - 'flag_bbox': example image bounding box flags as a torch tensor of shape M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated with the
                    given image, and N is the maximum number of annotations associated with a pair (image, class).
                - 'gt': query image classes mask as a tensor of shape H x W, in which each pixel has a certain value k if
                    that pixel is in the mask of the k-th class associated with the query image.
                - 'classes': dictionary in which each pair k: v represents the ith class corresponding to class id.

        Returns:
            Tuple[Dict[str, Any], torch.Tensor]: A tuple containing the batched dictionary and the batched output masks.
                The batched dictionary has the following entries:
                - 'query_image': query image as a torch tensor of shape B x 3 x H x W.
                - 'example_images': example images as a torch tensor of shape B x M x 3 x H x W.
                - 'point_coords': example image coordinates as a torch tensor of shape B x M x C x N x K x 2, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated with
                    the given image, N is the maximum number of annotations associated with a pair (image, class), and K
                    is the number of points extracted.
                - 'point_flags': example image coordinate flags as a torch tensor of shape B x M x C x N x K, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated with
                    the given image, N is the maximum number of annotations associated with a pair (image, class), and K
                    is the number of points extracted.
                - 'boxes': example image bounding boxes as a torch tensor of shape B x M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated with the
                    given image, and N is the maximum number of annotations associated with a pair (image, class). The
                    last dimension is 4 because a single bounding box is represented by the top-left and bottom-right
                    coordinates.
                - 'box_flags': example image bounding box flags as a torch tensor of shape B x M x C x N x 4, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated with
                    the given image, and N is the maximum number of annotations associated with a pair (image, class).
                - 'mask_inputs': example image masks as a torch tensor of shape B x M x C x H x W, where M is the number
                    of examples extracted for the given query image and C is the number of classes associated with it.
            The batched output masks is a torch tensor of shape B x H x W.
        """
        batched_input, dataset_names = zip(*batched_input)
        # classes
        max_classes = max([x["prompt_masks"].size(1) for x in batched_input])

        # gt
        dims = torch.stack([x["dims"] for x in batched_input])
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        ground_truths = [x["ground_truths"] for x in batched_input]
        ground_truths = torch.stack(
            [utils.collate_batch_gts(x, max_dims) for x in ground_truths]
        )

        # prompt mask
        masks = [x["prompt_masks"] for x in batched_input]
        flags = [x["flag_masks"] for x in batched_input]
        masks_flags = [
            utils.collate_mask(m, f, max_classes) for (m, f) in zip(masks, flags)
        ]
        masks = torch.stack([x[0] for x in masks_flags])
        flag_masks = torch.stack([x[1] for x in masks_flags])

        # prompt bbox
        bboxes = [x["prompt_bboxes"] for x in batched_input]
        flags = [x["flag_bboxes"] for x in batched_input]
        max_annotations = max(x.size(2) for x in bboxes)
        bboxes_flags = [
            utils.collate_bbox(bboxes[i], flags[i], max_classes, max_annotations)
            for i in range(len(bboxes))
        ]
        bboxes = torch.stack([x[0] for x in bboxes_flags])
        flag_bboxes = torch.stack([x[1] for x in bboxes_flags])

        # prompt coords
        points = [x["prompt_points"] for x in batched_input]
        flags = [x["flag_points"] for x in batched_input]
        max_annotations = max(x.size(2) for x in points)
        points_flags = [
            utils.collate_coords(points[i], flags[i], max_classes, max_annotations)
            for i in range(len(points))
        ]
        points = torch.stack([x[0] for x in points_flags])
        flag_points = torch.stack([x[1] for x in points_flags])

        # flag examples
        flag_examples = torch.stack(
            [utils.collate_example_flags(x["flag_examples"], max_classes) for x in batched_input]
        )

        # aux gts
        classes = [x["classes"] for x in batched_input]

        # image ids
        image_ids = [x["image_ids"] for x in batched_input]

        # flag_gts
        flag_gts = torch.zeros((len(batched_input), max_classes), dtype=torch.bool)
        for i, x in enumerate(classes):
            flag_gts[i, : len(list(set(itertools.chain(*x)))) + 1] = 1

        # images
        if "embeddings" in batched_input[0].keys():
            image_key = "embeddings"
            images = torch.stack([x[image_key] for x in batched_input])
        else:
            image_key = "images"
            images = torch.stack([x["images"] for x in batched_input])

        data_dict = {
            image_key: images,
            "prompt_points": points,
            "flag_points": flag_points,
            "prompt_bboxes": bboxes,
            "flag_bboxes": flag_bboxes,
            "prompt_masks": masks,
            "flag_masks": flag_masks,
            "flag_examples": flag_examples,
            "dims": dims,
            "classes": classes,
            "image_ids": image_ids,
            "flag_gts": flag_gts,
        }

        return (data_dict, ground_truths), dataset_names


def get_batch_metadata(
    dataset_len, possible_batch_example_nums, possible_prompts, num_processes=1
):
    """
    Returns a list of number of examples per batch and a list of batch sizes
    such that the total number of examples is `batch_size * max_num_examples`
    """
    examples_nums = []
    batch_sizes = []
    prompt_types = []
    num_classes = []
    combs = [
        list(itertools.combinations(possible_prompts, i))
        for i in range(1, len(possible_prompts) + 1)
    ]
    multi_combs = [x for comb in combs for x in comb]
    remaining_images = dataset_len // num_processes
    while remaining_images > 0:
        res = random.choice(possible_batch_example_nums)
        num_class = None
        if len(res) == 2:
            cur_batch_size, examples_num = res
        elif len(res) == 3:
            cur_batch_size, num_class, examples_num = res
        else:
            raise ValueError("Invalid number of elements in the batch metadata.")
        if cur_batch_size > remaining_images:
            cur_batch_size = remaining_images
        prompt_type = random.choice(multi_combs)
        prompt_types.append(prompt_type)
        examples_nums.append(examples_num)
        batch_sizes.append(cur_batch_size)
        if num_class is not None:
            num_classes.append(num_class)
        remaining_images -= cur_batch_size

    batch_sizes = [
        val for tup in zip(*[batch_sizes for i in range(num_processes)]) for val in tup
    ]
    examples_nums = [
        val
        for tup in zip(*[examples_nums for i in range(num_processes)])
        for val in tup
    ]
    prompt_types = [
        val for tup in zip(*[prompt_types for i in range(num_processes)]) for val in tup
    ]
    batch_metadata = {
        utils.BatchMetadataKeys.NUM_EXAMPLES: examples_nums,
        utils.BatchMetadataKeys.PROMPT_TYPES: prompt_types,
    }
    if len(num_classes) > 0:
        num_classes = [
            val for tup in zip(*[num_classes for i in range(num_processes)]) for val in tup
        ]
        batch_metadata[utils.BatchMetadataKeys.NUM_CLASSES] = num_classes

    return batch_sizes, batch_metadata


class VariableBatchSampler(BatchSampler):
    """
    A custom batch sampler that generates variable-sized batches based on the provided constraints.

    Args:
        data_source (Dataset): The dataset to sample from.
        max_batch_size (int): The maximum size of each batch.
        max_num_examples (int): The maximum number of examples to include in each batch.
        prompt_types (List[str]): The types of prompts to use.
        drop_last (bool, optional): Whether to drop the last batch if it is smaller than `max_batch_size`. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the data before sampling. Defaults to False.
        num_processes (int, optional): The number of processes to use for parallel processing. Defaults to 1.

    Raises:
        ValueError: If no batch size is provided.

    Returns:
        Iterator: An iterator that yields variable-sized batches.

    """

    def __init__(
        self,
        data_source,
        possible_batch_example_nums,
        prompt_types=None,
        drop_last=False,
        shuffle=False,
        num_processes=1,
        num_steps=None,
    ):
        self.data_source = data_source
        if prompt_types is None:
            prompt_types = [
                utils.PromptType.BBOX,
                utils.PromptType.MASK,
                utils.PromptType.POINT,
            ]

        self.batch_sizes, self.batch_metadata = get_batch_metadata(
            len(data_source),
            possible_batch_example_nums,
            num_processes=num_processes,
            possible_prompts=prompt_types,
        )
        self.num_processes = num_processes
        if num_steps is not None:
            if num_steps % num_processes != 0:
                logger.warning(
                    "The number of steps is not divisible by the number of processes."
                )
                logger.warning(
                    "The number of steps will be adjusted to be divisible by the number of processes."
                )
                num_steps = num_steps - (num_steps % num_processes)
                logger.warning(f"The new number of steps is {num_steps}.")
            self.batch_sizes = self.batch_sizes[:num_steps]
            self.batch_metadata = {
                k: v[:num_steps] for k, v in self.batch_metadata.items()
            }
        self.drop_last = drop_last
        self.do_shuffle = shuffle
        if shuffle:
            self.sampler = torch.utils.data.RandomSampler(data_source)
        else:
            self.sampler = torch.utils.data.SequentialSampler(data_source)

        if len(self.batch_sizes) == 0:
            raise ValueError("At least one batch size should be provided.")

    def __len__(self):
        return len(self.batch_sizes)
    
    def shuffle(self):
        # Remove th processes multiplication
        batches = self.batch_sizes[::self.num_processes]
        metadata = {k: v[::self.num_processes] for k, v in list(self.batch_metadata.items())}
        # Get permutation
        indices = torch.randperm(len(batches)).tolist()
        # Permute
        batches = [batches[i] for i in indices]
        # Multiply for the number of processes
        metadata = {k: [v[i] for i in indices] for k, v in metadata.items()}
        self.batch_sizes = [
            val for tup in zip(*[batches for i in range(self.num_processes)]) for val in tup
        ]
        self.batch_metadata = {
            k: list(val for tup in zip(*[v for i in range(self.num_processes)]) for val in tup)
            for k, v in metadata.items()
        }
        

    def __iter__(self):
        if self.do_shuffle:
            self.shuffle()
            indices = iter(list(torch.randperm(len(self.sampler.data_source)).tolist()))
        else:
            indices = self.sampler.__iter__()

        for i, batch_size in enumerate(self.batch_sizes):
            metadata = {k: v[i] for k, v in self.batch_metadata.items()}
            batch = []
            while len(batch) < batch_size and indices:
                batch.append((next(indices), metadata))
            yield batch

    def get_max_num_images(self):
        return self.batch_sizes[0] * self.batch_metadata[0]
