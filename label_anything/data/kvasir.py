from label_anything.data.test import LabelAnythingTestDataset
from label_anything.data.utils import BatchKeys
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
        img_dir: str,
        mask_dir: str,
        annotations: str,
        preprocess=None,
        prompt_images=None,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.annotations = annotations
        self.preprocess = preprocess
        if prompt_images is None:
            prompt_images = [
                "cju0qkwl35piu0993l0dewei2.jpg",
                "cjyzl833ndne80838pzuq6ila.jpg",
                "cju323ypb1fbb0988gx5rzudb.jpg",
                "cju88itqbny720987hxizbj5y.jpg",
                "cju87kbcen2av0987usezo8kn.jpg",
            ]
        self.prompt_images = prompt_images
        self.filenames = os.listdir(self.img_dir)

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def _transform_image(self, image):
        image = Image.fromarray(image.permute(1, 2, 0).numpy().astype("uint8"))
        image = self.preprocess(image)
        return image

    def _get_image(self, filename: str):
        return torchvision.io.read_image(os.path.join(self.img_dir, filename))

    def _get_gt(self, filename: str):
        mask = Image.open(os.path.join(self.mask_dir, filename))
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
            self._get_image(os.path.join(self.img_dir, filename))
            for filename in self.prompt_images
        ]
        images = [self._transform_image(image) for image in images]
        sizes = torch.stack([torch.tensor(image.shape[1:]) for image in images])
        masks = [
            self._get_gt(os.path.join(self.mask_dir, filename))
            for filename in self.prompt_images
        ]
        masks = [self._pad_mask(mask) for mask in masks]
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
