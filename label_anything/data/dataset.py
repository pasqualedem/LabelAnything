import itertools
import os
import random
import timeit
import warnings
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms
import utils
from examples import generate_examples_power_law_uniform
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    PILToTensor,
    Resize,
    ToTensor,
)
from transforms import CustomNormalize, CustomResize, PromptsProcessor

warnings.filterwarnings("ignore")


class PromptType(Enum):
    BBOX = 0
    MASK = 1
    POINT = 2


class LabelAnythingDataset(Dataset):
    def __init__(
        self,
        instances_path,  # Path
        img_dir=None,  # directory (only if images have to be loaded from disk)
        max_num_examples=10,  # number of max examples to be given for the target image
        preprocess=ToTensor(),  # preprocess step
        j_index_value=0.5,  # threshold for extracting examples
        seed=42,  # for reproducibility
        max_mum_coords=10,  # max number of coords for each example for each class
    ):
        super().__init__()
        instances = utils.load_instances(instances_path)
        self.load_from_dir = img_dir is not None
        self.img_dir = img_dir

        # id to annotation
        self.annotations = dict((x["id"], x) for x in instances["annotations"])
        # list of image ids
        self.image_ids = [x["id"] for x in instances["images"]]
        # id to image
        self.images = dict((x["id"], x) for x in instances["images"])
        # id to category
        self.categories = dict((x["id"], x) for x in instances["categories"])
        # img id to cat id to annotations
        # cat id to img id to annotations
        (
            self.img2cat_annotations,
            self.cat2img_annotations,
        ) = self.__load_annotation_dicts()

        self.max_num_examples = max_num_examples
        self.max_num_coords = max_mum_coords

        # assert that they are positive
        assert self.max_num_examples > 0
        assert self.max_num_coords > 0

        self.preprocess = preprocess
        self.prompts_processor = PromptsProcessor(
            long_side_length=1024, masks_side_length=256
        )

        self.j_index_value = j_index_value
        self.seed = seed
        self.reset_num_examples()
        self.reset_num_coords()

    def reset_num_examples(self):
        self.num_examples = random.randint(1, self.max_num_examples)

    def reset_num_coords(self):
        self.num_coords = random.randint(1, self.max_num_coords)

    def __load_annotation_dicts(self) -> (dict, dict):
        """Prepares dictionaries for fast access to annotations.

        Returns:
            (dict, dict): Returns two dictionaries:
                1. img2cat_annotations: image id to category id to annotations
                2. cat2img_annotations: category id to image id to annotations
        """
        img2cat_annotations = dict()
        cat2img_annotations = dict()

        for ann in self.annotations.values():
            if not ann["image_id"] in img2cat_annotations:
                img2cat_annotations[ann["image_id"]] = dict()
            if not ann["category_id"] in img2cat_annotations[ann["image_id"]]:
                img2cat_annotations[ann["image_id"]][ann["category_id"]] = []
            img2cat_annotations[ann["image_id"]][ann["category_id"]].append(ann)
            if not ann["category_id"] in cat2img_annotations:
                cat2img_annotations[ann["category_id"]] = dict()
            if not ann["image_id"] in cat2img_annotations[ann["category_id"]]:
                cat2img_annotations[ann["category_id"]][ann["image_id"]] = []
            cat2img_annotations[ann["category_id"]][ann["image_id"]].append(ann)

        return img2cat_annotations, cat2img_annotations

    def __load_image(self, img_data: dict) -> Image:
        """Load an image from disk or from url.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            PIL.Image: The loaded image.
        """
        if self.load_from_dir:
            return Image.open(
                f'{self.img_dir}/{img_data["file_name"]}'
            )  # probably needs to add zeroes
        return Image.open(BytesIO(requests.get(img_data["coco_url"]).content))

    def __extract_examples(self, img_data: dict) -> (list, list):
        """Chooses examples (and categories) for the query image.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            (list, list): Returns two lists:
                1. examples: A list of image ids of the examples.
                2. cats: A list of category ids of the examples.
        """

        return generate_examples_power_law_uniform(
            query_image_id=img_data["id"],
            image_classes=self.img2cat_annotations[img_data["id"]],
            categories_to_imgs=self.cat2img_annotations,
            min_size=1,
            num_examples=self.num_examples,
        )

    def __getitem__(self, item: int) -> dict:
        base_image_data = self.images[self.image_ids[item]]

        # load the examples and categories for the query image
        start = timeit.default_timer()
        image_ids, aux_cat_ids = self.__extract_examples(base_image_data)
        cat_ids = list(set(itertools.chain(*aux_cat_ids)))
        cat_ids.insert(0, -1)  # add the background class
        end = timeit.default_timer()
        print(f"Time to extract examples: {end - start}")

        # load, stack and preprocess the images
        start = timeit.default_timer()
        images = [
            self.__load_image(image_data)
            for image_data in [self.images[image_id] for image_id in image_ids]
        ]
        images = torch.stack(
            [
                image if not self.preprocess else self.preprocess(image)
                for image in images
            ],
            dim=0,
        )
        end = timeit.default_timer()
        print(f"Time to load and preprocess images: {end - start}")

        # create the prompt dicts
        bboxes = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}
        masks = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}
        points = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}

        # get prompts from annotations
        start = timeit.default_timer()
        for img_id in image_ids:
            img_size = (self.images[img_id]["height"], self.images[img_id]["width"])
            for cat_id in cat_ids:
                # for each annotation of image img_id and category cat_id
                if cat_id not in self.img2cat_annotations[img_id]:
                    # this will also manage the background class
                    continue
                for ann in self.img2cat_annotations[img_id][cat_id]:
                    # choose the prompt type
                    prompt_type = random.choice(list(PromptType))

                    if prompt_type == PromptType.BBOX:
                        # take the bbox
                        bboxes[img_id][cat_id].append(
                            self.prompts_processor.convert_bbox(ann["bbox"]),
                        )
                    elif prompt_type == PromptType.MASK:
                        # take the mask
                        masks[img_id][cat_id].append(
                            self.prompts_processor.convert_mask(
                                ann["segmentation"],
                                *img_size,
                            )
                        )
                    elif prompt_type == PromptType.POINT:
                        # take the point
                        mask = self.prompts_processor.convert_mask(
                            ann["segmentation"],
                            *img_size,
                        )
                        points[img_id][cat_id].append(
                            self.prompts_processor.sample_point(mask)
                        )

        # convert the lists of prompts to arrays
        for img_id in image_ids:
            for cat_id in cat_ids:
                bboxes[img_id][cat_id] = np.array((bboxes[img_id][cat_id]))
                masks[img_id][cat_id] = np.array((masks[img_id][cat_id]))
                points[img_id][cat_id] = np.array((points[img_id][cat_id]))
        end = timeit.default_timer()
        print(f"Time to extract prompts: {end - start}")

        # obtain padded tensors
        start = timeit.default_timer()
        bboxes, flag_bboxes = self.annotations_to_tensor(bboxes, PromptType.BBOX)
        masks, flag_masks = self.annotations_to_tensor(masks, PromptType.MASK)
        points, flag_points = self.annotations_to_tensor(points, PromptType.POINT)
        end = timeit.default_timer()
        print(f"Time to convert prompts to tensors: {end - start}")

        # obtain ground truths
        start = timeit.default_timer()
        ground_truths = self.get_ground_truths(image_ids, cat_ids)
        dims = torch.tensor(list(map(lambda x: x.size(), ground_truths)))
        max_dims = torch.max(dims, 0).values.tolist()
        ground_truths = torch.stack(
            [utils.collate_gts(x, max_dims) for x in ground_truths]
        )
        end = timeit.default_timer()
        print(f"Time to obtain ground truths: {end - start}")

        return {
            "images": images,
            "prompt_masks": masks,
            "flag_masks": flag_masks,
            "prompt_points": points,
            "flag_points": flag_points,
            "prompt_bboxes": bboxes,
            "flag_bboxes": flag_bboxes,
            "dims": dims,
            "classes": aux_cat_ids,
            "ground_truths": ground_truths,
        }

    def get_ground_truths(self, image_ids, cat_ids):
        # initialization
        ground_truths = dict((img_id, {}) for img_id in image_ids)
        # generate masks
        for img_id in image_ids:
            img_size = (self.images[img_id]["height"], self.images[img_id]["width"])
            for cat_id in cat_ids:
                ground_truths[img_id][cat_id] = np.zeros(img_size, dtype=np.uint8)
                # zero mask for no segmentation
                if cat_id not in self.img2cat_annotations[img_id]:
                    continue
                for ann in self.img2cat_annotations[img_id][cat_id]:
                    ground_truths[img_id][cat_id] = np.logical_or(
                        ground_truths[img_id][cat_id],
                        self.prompts_processor.convert_mask(
                            ann["segmentation"], *img_size
                        ),
                    )
            # make the ground truth tensor for image img_id
            ground_truth = torch.from_numpy(
                np.array([ground_truths[img_id][cat_id].astype(np.uint8) for cat_id in cat_ids])
            )
            # add a zeroes tensor to the first dimension
            ground_truth = torch.cat(
                [
                    torch.zeros((1, *ground_truth.shape[1:])).type(torch.uint8),
                    ground_truth,
                ]
            )
            ground_truths[img_id] = torch.argmax(ground_truth, 0)

        return list(ground_truths.values())

    def __len__(self):
        return len(self.images)

    def annotations_to_tensor(self, annotations, prompt_type) -> torch.Tensor:
        """Transform a dict of annotations of prompt_type to a padded tensor.

        Args:
            annotations (dict): annotations (dict of dicts with np.ndarray as values)
            prompt_type (PromptType): prompt type

        Returns:
            torch.Tensor: padded tensor
        """
        n = len(annotations)
        c = len(next(iter(annotations.values())))

        m = 10  # max number of annotations by type
        if prompt_type == PromptType.BBOX:
            max_annotations = utils.get_max_annotations(annotations)
            tensor_shape = (n, c, max_annotations, 4)
        elif prompt_type == PromptType.MASK:
            tensor_shape = (n, c, 256, 256)
        elif prompt_type == PromptType.POINT:
            max_annotations = utils.get_max_annotations(annotations)
            tensor_shape = (n, c, max_annotations, 2)

        tensor = torch.zeros(tensor_shape)
        flag = (
            torch.zeros(tensor_shape[:-1]).type(torch.uint8)
            if prompt_type != PromptType.MASK
            else torch.zeros(tensor_shape[:2]).type(torch.uint8)
        )

        if prompt_type == PromptType.MASK:
            for i, img_id in enumerate(annotations):
                for j, cat_id in enumerate(annotations[img_id]):
                    mask = self.prompts_processor.apply_masks(
                        annotations[img_id][cat_id]
                    )
                    tensor_mask = torch.tensor(mask)
                    tensor[i, j, :] = tensor_mask
                    flag[i, j] = 1 if torch.sum(tensor_mask) > 0 else 0
        else:
            for i, img_id in enumerate(annotations):
                img_original_size = (
                    self.images[img_id]["height"],
                    self.images[img_id]["width"],
                )
                for j, cat_id in enumerate(annotations[img_id]):
                    if annotations[img_id][cat_id].size == 0:
                        continue
                    m = annotations[img_id][cat_id].shape[0]
                    if prompt_type == PromptType.BBOX:
                        boxes_ann = self.prompts_processor.apply_boxes(
                            annotations[img_id][cat_id], img_original_size
                        )
                        tensor[i, j, :m, :] = torch.tensor(boxes_ann)
                    elif prompt_type == PromptType.POINT:
                        points_ann = self.prompts_processor.apply_coords(
                            annotations[img_id][cat_id], img_original_size
                        )
                        tensor[i, j, :m, :] = torch.tensor(points_ann)
                    flag[i, j, :m] = 1

        return tensor, flag

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
        # classes
        max_classes = max([x["prompt_masks"].size(1) for x in batched_input])

        # gt
        dims = torch.stack([x["dims"] for x in batched_input])
        max_dims = torch.max(dims, 0).values.tolist()
        ground_truths = [x["query_gt"] for x in batched_input]
        ground_truths = torch.stack(
            [utils.collate_gts(x, max_dims) for x in ground_truths]
        )

        # prompt mask
        masks = [x["prompt_masks"] for x in batched_input]
        flags = [x["flag_masks"] for x in batched_input]
        masks_flags = [
            utils.collate_mask(m, f, max_classes) for (m, f) in zip(masks, flags)
        ]
        masks = torch.stack([x[0] for x in masks_flags])
        mask_flags = torch.stack([x[1] for x in masks_flags])

        # prompt bbox
        bboxes = [x["prompt_bboxes"] for x in batched_input]
        flags = [x["flag_bboxes"] for x in batched_input]
        max_annotations = max(x.size(2) for x in bboxes)
        bboxes_flags = [
            utils.collate_bbox(bboxes[i], flags[i], max_classes, max_annotations)
            for i in range(len(bboxes))
        ]
        bboxes = torch.stack([x[0] for x in bboxes_flags])
        bbox_flags = torch.stack([x[1] for x in bboxes_flags])

        # prompt coords
        coords = [x["prompt_coords"] for x in batched_input]
        flags = [x["flag_coords"] for x in batched_input]
        max_annotations = max(x.size(2) for x in coords)
        coords_flags = [
            utils.collate_coords(coords[i], flags[i], max_classes, max_annotations)
            for i in range(len(coords))
        ]
        coords = torch.stack([x[0] for x in coords_flags])
        coord_flags = torch.stack([x[1] for x in coords_flags])

        # query image
        query_image = torch.stack([x["query_image"] for x in batched_input])

        # example image
        example_images = torch.stack([x["examples"] for x in batched_input])
        example_dims = torch.stack([x["example_dim"] for x in batched_input])
        max_example_dims = torch.max(example_dims.view(-1, 2), 0).values.tolist()
        example_gt = [x["example_gts"] for x in batched_input]
        example_gt = torch.stack(
            [utils.collate_example_gts(x, max_example_dims) for x in example_gt]
        )

        data_dict = {
            "query_image": query_image,
            "example_images": example_images,
            "point_coords": coords,
            "point_flags": coord_flags,
            "boxes": bboxes,
            "box_flags": bbox_flags,
            "mask_inputs": masks,
            "mask_flags": mask_flags,
            "gt_dims": dims,
            "example_dims": example_dims,
            "example_gts": example_gt,
        }

        # reset dataset parameters
        self.reset_num_coords()
        self.reset_num_examples()

        return data_dict, ground_truths


class LabelAnyThingOnlyImageDataset(Dataset):
    def __init__(self, directory=None, preprocess=None):
        super().__init__()
        self.directory = directory
        self.files = os.listdir(directory)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.directory, self.files[item]))
        image_id, _ = os.path.splitext(self.files[item])
        return self.preprocess(img), image_id  # load image


# main for testing the class
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor

    preprocess = Compose(
        [
            CustomResize(1024),
            PILToTensor(),
            CustomNormalize(),
        ]
    )

    dataset = LabelAnythingDataset(
        instances_path="data/raw/lvis_v1_train.json",
        preprocess=preprocess,
        max_num_examples=10,
        j_index_value=0.1,
    )

    """x = dataset[1]
    print([f'{k}: {v.size()}' for k, v in x.items() if isinstance(v, torch.Tensor)])
    exit()"""

    dataloader = DataLoader(
        dataset=dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn
    )
    data_dict, gt = next(iter(dataloader))

    print([f"{k}: {v.size()}" for k, v in data_dict.items()])
    print(f"gt: {gt.size()}")
