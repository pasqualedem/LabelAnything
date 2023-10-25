from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import random
import torch
from torchvision.transforms import Resize, ToTensor, InterpolationMode, PILToTensor
import warnings
import utils
from transforms import (
    CustomResize,
    CustomNormalize,
    PromptsProcessor,
)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

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
            long_side_length=1024,
            masks_side_length=256
        )

        self.j_index_value = j_index_value
        self.seed = seed

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
        cats = list(self.img2cat_annotations[img_data["id"]].keys())
        cats = cats[:2]
        examples = [
            set(self.cat2img_annotations[cats[0]].keys()),
            set(self.cat2img_annotations[cats[1]].keys()),
        ]
        examples = list(examples[0].intersection(examples[1]))

        return examples[:10], cats

    def __getitem__(self, item: int) -> dict:
        image_data = self.images[self.image_ids[item]]
 
        # load (and preprocess) the query image
        query_image = self.__load_image(image_data)
        query_image = (
            query_image if not self.preprocess else self.preprocess(query_image)
        )

        # load the examples and categories for the query image
        example_ids, cat_ids = self.__extract_examples(image_data)
        examples = [
            self.__load_image(example_data)
            for example_data in [self.images[example_id] for example_id in example_ids]
        ]
        examples = torch.stack([
            example if not self.preprocess else self.preprocess(example)
            for example in examples
        ], dim=0)

        # create the prompt dicts
        bboxes = dict((img_id, {}) for img_id in example_ids)
        masks = dict((img_id, {}) for img_id in example_ids)
        points = dict((img_id, {}) for img_id in example_ids)

        # initialize the prompt dicts
        for img_id in example_ids:
            for cat_id in cat_ids:
                bboxes[img_id][cat_id] = []
                masks[img_id][cat_id] = []
                points[img_id][cat_id] = []

        # get prompts from examples' annotations
        for img_id in example_ids:
            img_size = (self.images[img_id]["height"], self.images[img_id]["width"])
            for cat_id in cat_ids:
                # for each annotation of image img_id and category cat_id
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
        for img_id in example_ids:
            for cat_id in cat_ids:
                bboxes[img_id][cat_id] = np.array((bboxes[img_id][cat_id]))
                masks[img_id][cat_id] = np.array((masks[img_id][cat_id]))
                points[img_id][cat_id] = np.array((points[img_id][cat_id]))

        # obtain padded tensors
        bboxes, flag_bbox = self.annotations_to_tensor(bboxes, PromptType.BBOX)
        masks,  flag_masks = self.annotations_to_tensor(masks, PromptType.MASK)
        points, flag_points = self.annotations_to_tensor(points, PromptType.POINT)

        # gt
        query_masks = dict((cat_id, []) for cat_id in cat_ids)
        for cat_id in cat_ids:
            for ann in self.img2cat_annotations[image_data["id"]][cat_id]:
                query_masks[cat_id].append(
                    self.prompts_processor.convert_mask(
                        ann["segmentation"], image_data["height"], image_data["width"]
                    )
                )
        query_gt = torch.as_tensor(
            [
                np.logical_or.reduce(query_masks[cat_id]).astype(np.uint8)
                for cat_id in cat_ids
            ]
        )
        query_gt = torch.argmax(query_gt, 0)

        dims = torch.Tensor(query_gt.size()).type(torch.uint8)

        return {
            "query_image": query_image,  # 3 x H x W
            "examples": examples,  # N x 3 x H x W
            "prompt_mask": masks,  # N x C x H_M x W_M
            "flag_masks": flag_masks,
            "prompt_coords": points, # N x C x M x 2
            "flag_coords": flag_points,
            "prompt_bboxes": bboxes, # N x C x M x 4
            "flag_bboxes": flag_bbox,
            "query_gt": query_gt,  # C x H x W
            "dims": dims,
        }

    def __len__(self):
        return len(self.images)

    # TODO: add max_annotations and mask shape
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

        m = 10 # max number of annotations by type
        if prompt_type == PromptType.BBOX:
            tensor_shape = (n, c, 10, 4)
        elif prompt_type == PromptType.MASK:
            tensor_shape = (n, c, 256, 256)
        elif prompt_type == PromptType.POINT:
            tensor_shape = (n, c, 10, 2)

        tensor = torch.zeros(tensor_shape)
        flag = torch.zeros(tensor_shape[:-1]).type(torch.uint8)

        if prompt_type == PromptType.MASK:
            for i, img_id in enumerate(annotations):
                for j, cat_id in enumerate(annotations[img_id]):
                    mask = self.prompts_processor.apply_masks(
                        annotations[img_id][cat_id]
                    )
                    tensor[i, j, :] = torch.tensor(mask)
                    flag[i, j] = 1
        else:
            for i, img_id in enumerate(annotations):
                for j, cat_id in enumerate(annotations[img_id]):
                    if annotations[img_id][cat_id].size == 0:
                        continue
                    m = annotations[img_id][cat_id].shape[0]
                    tensor[i, j, :m, :] = torch.tensor(annotations[img_id][cat_id])
                    flag[i, j, :m] = 1

        return tensor, flag


# main for testing the class
if __name__ == "__main__":
    from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
    from torch.utils.data import DataLoader

    preprocess = Compose(
        [
            CustomResize(1024),
            PILToTensor(),
            CustomNormalize(),
        ]
    )

    dataset = LabelAnythingDataset(
        instances_path="lvis_v1_train.json",
        preprocess=preprocess,
        max_num_examples=10,
        j_index_value=0.1,
    )
    
    x = dataset[1]
    for k, v in x.items():
        print(f'{k}: {v.size()}')
