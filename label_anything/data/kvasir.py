from label_anything.data.test import LabelAnythingTestDataset
from label_anything.data.utils import (
    BatchKeys,
    PromptType,
    flags_merge,
    annotations_to_tensor,
)
import os
from PIL import Image
import torchvision
import numpy as np
import json
import torch
from torchvision import transforms
from pycocotools import mask as mask_utils
from torch.nn.functional import one_hot

import torch.nn.functional as F


class KvarisTestDataset(LabelAnythingTestDataset):
    id2class = {0: "background", 1: "polyp"}
    num_classes = 2

    def __init__(
        self,
        root: str,
        preprocess=None,
        prompt_images=None,
    ):
        super().__init__()
        self.root = root
        self.test_root = os.path.join(self.root, "test")
        self.train_root = os.path.join(self.root, "train")
        self.annotations = self._read_bbox()
        self.preprocess = preprocess
        if prompt_images is None:
            prompt_images = [
                "cju0qkwl35piu0993l0dewei2.jpg",
                "cju0t4oil7vzk099370nun5h9.jpg",
                "cju323ypb1fbb0988gx5rzudb.jpg",
                # "cju1cj3f0qi5n0993ut8f49rj.jpg",
                # "cju772304yw5t0818vbw8kkjf.jpg",
            ]
        self.prompt_images = prompt_images
        self.filenames = os.listdir(os.path.join(self.test_root, "images"))

    def __len__(self):
        return len(os.listdir(os.path.join(self.test_root, "images")))

    def _read_bbox(self):
        with open(os.path.join(self.root, "kavsir_bboxes.json")) as f:
            data = json.load(f)
        return data

    def _get_bbox(self, json_data, filename: str):
        for k, v in json_data.items():
            if k == filename.split(".")[0]:
                return torch.tensor(
                    [
                        v.get("bbox")[0].get("xmin"),
                        v.get("bbox")[0].get("ymin"),
                        v.get("bbox")[0].get("xmax"),
                        v.get("bbox")[0].get("ymax"),
                    ]
                )

    def _transform_image(self, image):
        image = Image.fromarray(image.permute(1, 2, 0).numpy().astype("uint8"))
        image = self.preprocess(image)
        return image

    def _get_image(self, filename: str):
        return torchvision.io.read_image(
            os.path.join(self.test_root, "images", filename)
        )

    def _get_gt(self, filename: str):
        mask = Image.open(os.path.join(self.test_root, "masks", filename))
        mask = torchvision.transforms.PILToTensor()(mask)[0]
        mask[mask <= 245] = 0
        mask[mask >= 245] = 1
        return mask.long()

    def __getitem__(self, idx):
        image = self._get_image(self.filenames[idx])
        size = torch.tensor(image.shape[1:])
        gt = self._get_gt(self.filenames[idx])
        image = self._transform_image(image)
        return {BatchKeys.IMAGES: image.unsqueeze(0), BatchKeys.DIMS: size}, gt

    def extract_prompts(self):
        images = [
            self._get_image(os.path.join(self.train_root, "images", filename))
            for filename in self.prompt_images
        ]
        bboxes = [
            self._get_bbox(self.annotations, filename)
            for filename in self.prompt_images
        ]
        images = [self._transform_image(image) for image in images]
        sizes = torch.stack([torch.tensor(image.shape[1:]) for image in images])
        masks = [
            self._get_gt(os.path.join(self.train_root, "masks", filename))
            for filename in self.prompt_images
        ]
        masks = [self._pad_mask(mask) for mask in masks]

        bboxes = torch.stack(bboxes)
        images = torch.stack(images)
        masks = torch.stack(masks)

        # Create flag masks
        backflag = torch.zeros(masks.shape[0])
        contain_polyp = (masks == 1).sum(dim=(1, 2)) > 0
        flag_masks = torch.stack([backflag, contain_polyp]).T
        masks = one_hot(masks.long(), self.num_classes).permute(0, 3, 1, 2).float()
        flag_examples = flag_masks.clone().bool()

        prompt_dict = {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.PROMPT_BBOXES: torch.zeros(*flag_examples.shape, 0, 4),
            BatchKeys.FLAG_BBOXES: torch.zeros(*flag_examples.shape, 0),
            BatchKeys.PROMPT_POINTS: torch.zeros(*flag_examples.shape, 0, 2),
            BatchKeys.FLAG_POINTS: torch.zeros(*flag_examples.shape, 0),
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict

    def _pad_mask(self, mask):
        return F.pad(mask, (0, 1024 - mask.shape[1], 0, 1024 - mask.shape[0]))
