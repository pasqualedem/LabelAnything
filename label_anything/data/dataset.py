import itertools
import os
import random
import warnings
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, ToTensor

import label_anything.data.utils as utils
from label_anything.data.coco import CocoLVISDataset
from label_anything.data.examples import ExampleGeneratorPowerLawUniform

datasets = {
    "coco": CocoLVISDataset,
    "lvis": CocoLVISDataset,
    "val_coco": CocoLVISDataset,
    "val_lvis": CocoLVISDataset,
    "ade20k": None,
}


class LabelAnythingDataset(Dataset):
    def __init__(self, datasets_params: Dict, common_params: Dict) -> None:
        self._log_images = True  # logs the first batch
        self.num_examples = 0
        self.max_num_examples = common_params.get("max_num_examples")
        self.load_embeddings = common_params.get("load_embeddings")
        self.do_subsample = common_params.get("do_subsample")
        self.add_box_noise = common_params.get("add_box_noise")

        self.datasets = {
            dataset_name: datasets[dataset_name](**{**common_params, **params})
            for dataset_name, params in datasets_params.items()
        }
        self.categories = {
            dataset_name: dataset.categories for dataset_name, dataset in self.datasets.items()
        }
        index = sum(
            [
                [(dataset_name, i) for i in range(len(dataset))]
                for dataset_name, dataset in self.datasets.items()
            ],
            [],
        )
        self.index = {i: index for i, index in enumerate(index)}

        self.reset_num_examples()
        super().__init__()

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets.values()])

    def __getitem__(self, index) -> Any:
        dataset_name, dataset_index = self.index[index]
        return self.datasets[dataset_name][dataset_index], dataset_name

    def reset_num_examples(self):
        """Set the number of examples for the next query image."""
        self.num_examples = random.randint(1, self.max_num_examples)
        for dataset in self.datasets.values():
            dataset.num_examples = self.num_examples

    @property
    def log_images(self):
        return self._log_images

    @log_images.setter
    def log_images(self, value):
        self._log_images = value
        for dataset in self.datasets.values():
            dataset.log_images = value

    def collate_fn(
        self, batched_input: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Performs the collate_fn, which is useful for batching data points in a dataloader.

        Arguments:
            batched_input: list of batch_size elements, in which each element is a dict with the following entries:
                'target': query image as a torch tensor of shape 3 x H x W.
                'examples': example image as a torch tensor of shape M x 3 x H x W, where M is the number of examples
                    extracted for the given query image.
                'prompt_mask': example image masks as a torch tensor of shape M x C x H x W, where M is the number of
                    examples extracted for the given query image and C is the number of classed associated to it.
                'prompt_coords': example image coordinates as a torch tensor of shape M x C x N x K x 2, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, N is the maximum number of annotations associated to a pair (image, class), and K is
                    the number of points extracted.
                'flag_coords': example image coordinate flags as a torch tensor of shape M x C x N x K, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, N is the maximum number of annotations associated to a pair (image, class), and K is
                    the number of points extracted.
                'prompt_bbox': example image bounding boxes as a torch tensor of shape M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, and N is the maximum number of annotations associated to a pair (image, class). The
                    last dimension is 4 because a single bounding box is represented by the top-left and bottom-right
                    coordinates.
                'flag_bbox': example image bounding box flags as a torch tensor of shape M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, and N is the maximum number of annotations associated to a pair (image, class).
                'gt': query image classes mask as a tensor of shape H x W, in which each pixel has a certain value k if
                    that pixel is in the mask of the k-th class associated to the query image.
                'classes': dict in which each pair k: v is ith class corresponding to class id.

        Returns:
            Dict[str, Any]: batched dictionary having the following entries:
                'query_image': query image as a torch tensor of shape B x 3 x H x W.
                'example_images': example images as a torch tensor of shape B x M x 3 x H x W.
                'point_coords':  example image coordinates as a torch tensor of shape B x M x C x N x K x 2, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated to
                    the given image, N is the maximum number of annotations associated to a pair (image, class), and K
                    is the number of points extracted.
                'point_flags': example image coordinate flags as a torch tensor of shape B xM x C x N x K, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated to
                    the given image, N is the maximum number of annotations associated to a pair (image, class), and K
                    is the number of points extracted.
                'boxes': example image bounding boxes as a torch tensor of shape B x M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, and N is the maximum number of annotations associated to a pair (image, class). The
                    last dimension is 4 because a single bounding box is represented by the top-left and bottom-right
                    coordinates.
                'box_flags': example image bounding box flags as a torch tensor of shape B x M x C x N x 4, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated to
                    the given image, and N is the maximum number of annotations associated to a pair (image, class).
                'mask_inputs': example image masks as a torch tensor of shape B x M x C x H x W, where M is the number
                    of examples extracted for the given query image and C is the number of classed associated to it.
            torch.Tensor: batched output masks as a torch tensor of shape B x H x W.

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

        # aux gts
        classes = [x["classes"] for x in batched_input]

        # flag_gts
        flag_gts = torch.zeros((len(batched_input), max_classes), dtype=torch.bool)
        for i, x in enumerate(classes):
            flag_gts[i, : len(x[0]) + 1] = 1

        # images
        if "embeddings" in batched_input[0].keys():
            image_key = "embeddings"
            images = torch.stack([x[image_key] for x in batched_input])
        else:
            image_key = "images"
            images = torch.stack([torch.stack(x["images"]) for x in batched_input])

        data_dict = {
            image_key: images,
            "prompt_points": points,
            "flag_points": flag_points,
            "prompt_bboxes": bboxes,
            "flag_bboxes": flag_bboxes,
            "prompt_masks": masks,
            "flag_masks": flag_masks,
            "dims": dims,
            "classes": classes,
            "flag_gts": flag_gts,
        }

        if self._log_images and self.load_embeddings:
            log_images = torch.stack([x["images"] for x in batched_input])
            data_dict["images"] = log_images

        # reset dataset parameters
        self.reset_num_examples()

        return (data_dict, ground_truths), dataset_names
