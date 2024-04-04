import json
import os
import random
import cv2
from pycocotools import mask as mask_utils
from typing import Optional
import xml.etree.ElementTree as ET
from PIL import Image
from torch.nn.functional import one_hot
import numpy as np
import torch
from scipy.ndimage import label, binary_dilation
from label_anything.data.coco20i import Coco20iDataset
from safetensors.torch import load_file

from label_anything.data.utils import AnnFileKeys, BatchKeys, PromptType
from label_anything.data.test import LabelAnythingTestDataset


class VOC5i(Coco20iDataset):

    def __init__(
        self,
        root: str,
        annotations: str,
        mask_folders: list,
        preprocess=None,
    ):
        super().__init__()
        self.root = root
        self.annotations = annotations
        self.mask_folders = mask_folders
        self.preprocess = preprocess

    def __len__(self):
        self.ids = []
        with open(self.annotations) as f:
            for line in f:
                image_path, _ = line.rstrip().split(" ")
                image_id = os.path.splitext(os.path.basename(image_path))[0]
                self.ids.append(image_id)
        return len(self.ids)

    def _get_label(self, image_id):
        annotation_file = os.path.join(self.root, "Annotations", image_id + ".xml")
        objects = ET.parse(annotation_file).findall("object")
        labels = []

        for object in objects:
            class_name = object.find("name").text.lower().strip()
            labels.append(class_name)

        return np.array(labels)

    def _get_images(self, image_id):
        image = Image.open(os.path.join(self.root, "JPEGImages", image_id + ".jpg"))
        size = image.size
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), torch.tensor(size).unsqueeze(0)

    def _get_masks(self, image_id):
        mask_dir = random.sample(
            [
                os.path.join(self.root, "SegmentationClass"),
                os.path.join(self.root, "SegmentationClassAug"),
            ],
            2,
        )
        for dir in mask_dir:
            mask_path = os.path.join(dir, image_id + ".png")
            if os.path.isfile(mask_path):
                break
            else:
                continue

        mask_array = np.array(Image.open(mask_path))
        unique_values = np.unique(mask_array)
        masks = {}

        for value in unique_values:
            if value not in [0, 255]:
                # Apply binary dilation before finding connected components
                dilated_mask = binary_dilation(mask_array == value)
                labeled_array, num_features = label(dilated_mask)
                for i in range(1, num_features + 1):
                    for i in range(1, num_features + 1):
                        mask = np.where(labeled_array == i, 1, 0)
                        rle = mask_utils.encode(
                            np.asfortranarray(mask.astype(np.uint8))
                        )
                        rle["counts"] = rle["counts"].decode(
                            "utf-8"
                        )  # Convert bytes to string
                        rle_mask_key = f"{value}_{i}"
                        masks[rle_mask_key] = rle

        return masks

    def __getitem__(self, idx):
        image, size = self._get_images(self.img_folder, self.ids[idx])
        gt = self._get_masks(self.mask_folder, self.ids[idx])

        if self.preprocess:
            gt = torch.from_numpy(gt).long()

        return {
            BatchKeys.IMAGES: image,
            BatchKeys.DIMS: size,
        }, gt

    def _load_safe(self, img_data: dict) -> (torch.Tensor, Optional[torch.Tensor]):
        """Open a safetensors file and load the embedding and the ground truth.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            (torch.Tensor, Optional[torch.Tensor]): Returns a tuple containing the embedding and the ground truth.
        """
        assert self.emb_dir is not None, "emb_dir must be provided."
        gt = None

        f = load_file(f"{self.emb_dir}/{str(img_data[AnnFileKeys.ID])}.safetensors")
        embedding = f["embedding"]
        if self.load_gts:
            gt = f[f"{self.name}_gt"]
        return embedding, gt


class PascalTestDataset(LabelAnythingTestDataset):
    num_classes = 20

    def __init__(
        self,
        root: str,
        instances_path: str,
        mask_folder: list,
        preprocess=None,
    ):
        super().__init__()
        self.root = root
        self.instances_path = instances_path
        self.mask_folder = mask_folder
        self.preprocess = preprocess

    def __len__(self):
        self.ids = []
        with open(self.instances_path) as f:
            for line in f:
                image_path, _ = line.rstrip().split(" ")
                image_id = os.path.splitext(os.path.basename(image_path))[0]
                self.ids.append(image_id)
        return len(self.ids)

    def _get_label(self, image_id):
        annotation_file = os.path.join(self.root, "Annotations", image_id + ".xml")
        objects = ET.parse(annotation_file).findall("object")
        labels = []

        for object in objects:
            class_name = object.find("name").text.lower().strip()
            labels.append(class_name)

        return np.array(labels)

    def _get_images(self, image_id):
        image = Image.open(os.path.join(self.root, "JPEGImages", image_id + ".jpg"))
        size = image.size
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), torch.tensor(size).unsqueeze(0)

    def _get_masks(self, image_id):
        mask_dir = random.sample(
            [
                os.path.join(self.root, "SegmentationClass"),
                os.path.join(self.root, "SegmentationClassAug"),
            ],
            2,
        )
        for dir in mask_dir:
            mask_path = os.path.join(dir, image_id + ".png")
            if os.path.isfile(mask_path):
                break
            else:
                continue

        mask_array = np.array(Image.open(mask_path))
        unique_values = np.unique(mask_array)
        masks = {}

        for value in unique_values:
            if value not in [0, 255]:
                # Apply binary dilation before finding connected components
                dilated_mask = binary_dilation(mask_array == value)
                labeled_array, num_features = label(dilated_mask)
                for i in range(1, num_features + 1):
                    for i in range(1, num_features + 1):
                        mask = np.where(labeled_array == i, 1, 0)
                        rle = mask_utils.encode(
                            np.asfortranarray(mask.astype(np.uint8))
                        )
                        rle["counts"] = rle["counts"].decode(
                            "utf-8"
                        )  # Convert bytes to string
                        rle_mask_key = f"{value}_{i}"
                        masks[rle_mask_key] = rle

        return masks

    def __getitem__(self, idx):
        image, size = self._get_images(self.img_folder, self.ids[idx])
        gt = self._get_masks(self.mask_folder, self.ids[idx])

        if self.preprocess:
            gt = torch.from_numpy(gt).long()

        return {
            BatchKeys.IMAGES: image,
            BatchKeys.DIMS: size,
        }, gt


if __name__ == "__main__":
    from label_anything.data.transforms import CustomNormalize, CustomResize
    from torchvision import transforms

    root = "/home/emanuele/LabelAnything/data/raw/VOCdevkit/VOC2012"
    mask_folders = [
        "/home/emanuele/LabelAnything/data/raw/VOCdevkit/VOC2012/SegmentationClass",
        "/home/emanuele/LabelAnything/data/raw/VOCdevkit/VOC2012/SegmentationClassAug",
    ]
    instances_path = "/home/emanuele/LabelAnything/data/raw/VOCdevkit/VOC2012/ImageSets/SegmentationAug/trainval_aug.txt"

    preprocess = transforms.Compose(
        [
            CustomResize(1024),
            transforms.PILToTensor(),
            CustomNormalize(),
        ]
    )
    dataset = VOC5i(instances_path, mask_folders, preprocess=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(dataset))
    print(next(iter(dataloader)))


class PascalVOCTestDataset(LabelAnythingTestDataset):
    num_classes = 20

    def __init__(
        self,
        instances_path: str,
        img_dir: str,
        preprocess=None,
    ):
        super().__init__()
        with open(instances_path, "r") as f:
            instances_path = json.load(f)
        self.instances_path = instances_path
        self.img_dir = img_dir  # data/raw/VOCdevkit/VOC2012
        self.preprocess = preprocess
        self.image_to_category = self._image_to_category()

    def _image_to_category(self):
        image_to_category = {}
        for annotation in self.instances_path["annotations"]:
            image_id = annotation["image_id"]
            category_id = annotation["category_id"]
            image_to_category[image_id] = category_id
        return image_to_category

    def __len__(self):
        return len(self.instances_path["images"])

    def _get_image(self, image_info):
        image_path = os.path.join(self.img_dir, image_info["coco_url"])
        img = Image.open(image_path)
        size = img.size
        if self.preprocess:
            img = self.preprocess(img)  # 3 x h x w
        return img, torch.tensor(size).unsqueeze(0)

    def _get_gt(self, annotation_info):
        gt = mask_utils.decode(annotation_info["segmentation"])
        if self.preprocess:
            gt = torch.from_numpy(gt)
        return gt.long()

    def __getitem__(self, idx):
        image_info = self.instances_path["images"][idx]
        annotation_info = self.instances_path["annotations"][idx]
        image, size = self._get_image(image_info)
        gt = self._get_gt(annotation_info)
        return {
            BatchKeys.IMAGES: image,
            BatchKeys.DIMS: size,
        }, gt


def extract_prompts(self):
    cat_images = [img for img, cat_id in self.image_to_category.items() if cat_id == 1]
    selected_images = random.sample(cat_images, min(5, len(cat_images)))

    # Get image data
    image_data = [self._get_image_by_id(image_id) for image_id in selected_images]
    masks = [self._get_gt_by_id(image_id) for image_id in selected_images]
    images, sizes = zip(*image_data)
    images = torch.stack(images)
    sizes = torch.stack(sizes)
    masks = torch.stack(masks)

    # Create flag masks
    backflag = torch.zeros(masks.shape[0])
    contain_tumor = (masks == 1).sum(dim=(1, 2)) > 0
    flag_masks = torch.stack([backflag, contain_tumor]).T
    masks = one_hot(masks.long(), 2).permute(0, 3, 1, 2).float()

    prompt_dict = {
        BatchKeys.IMAGES: images,
        BatchKeys.PROMPT_MASKS: masks,
        BatchKeys.FLAG_MASKS: flag_masks,
        BatchKeys.DIMS: sizes,
    }
    return prompt_dict


# if __name__ == "__main__":
#     from label_anything.data.transforms import CustomNormalize, CustomResize
#     from torchvision import transforms

#     instances_path = (
#         "/home/emanuele/LabelAnything/data/annotations/instances_voc12.json"
#     )
#     img_dir = "/home/emanuele/LabelAnything/data/raw/VOCdevkit/VOC2012"
#     preprocess = transforms.Compose(
#         [
#             CustomResize(1024),
#             transforms.PILToTensor(),
#             CustomNormalize(),
#         ]
#     )
#     dataset = PascalVOCTestDataset(instances_path, img_dir, preprocess=preprocess)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
#     print(len(dataset))
#     print(next(iter(dataloader)))
