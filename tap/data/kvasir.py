from tap.data.test import LabelAnythingTestDataset
from tap.data.utils import (
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
                # "cju0qx73cjw570799j4n5cjze.jpg",
                # "cju175facms5f0993a5tjikvt.jpg",
                # "cju323ypb1fbb0988gx5rzudb.jpg", 
                # "cju0ue769mxii08019zqgdbxn.jpg",
                "cju1euuc65wm00799m4sjdnnn.jpg", # o questo
                # "cju1hirfi7ekp0855q0vgm9qq.jpg",
                # "cju2i03ptvkiu0799xbbd4det.jpg",
                # "cju1gv7106qd008784gk603mg.jpg",
                # "cju2txjfzv60w098839dcimys.jpg",
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
        masks = [self._resize_mask(mask) for mask in masks]

        bboxes = torch.stack(bboxes)
        images = torch.stack(images)
        masks = torch.stack(masks)

        # Create flag masks
        backflag = torch.zeros(masks.shape[0])
        contain_polyp = (masks == 1).sum(dim=(1, 2)) > 0
        flag_masks = torch.stack([backflag, contain_polyp]).T
        masks = one_hot(masks.long(), self.num_classes).permute(0, 3, 1, 2).float()

        prompt_bboxes = torch.zeros(*flag_masks.shape, 1, 4)
        flag_bboxes = torch.zeros(*flag_masks.shape, 1)
        prompt_points = torch.zeros(*flag_masks.shape, 0, 2)
        flag_points = torch.zeros(*flag_masks.shape, 0)

        flag_examples = flags_merge(flag_masks, flag_points, flag_bboxes)

        prompt_dict = {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.PROMPT_BBOXES: prompt_bboxes,
            BatchKeys.FLAG_BBOXES: flag_bboxes,
            BatchKeys.PROMPT_POINTS: prompt_points,
            BatchKeys.FLAG_POINTS: flag_points,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict

    def _pad_mask(self, mask):
        return F.pad(mask, (0, 1024 - mask.shape[1], 0, 1024 - mask.shape[0]))

    def _resize_mask(self, mask):
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        mask = F.interpolate(mask, size=(256, 256), mode="nearest")
        mask = mask.squeeze(0).squeeze(0)
        return mask
