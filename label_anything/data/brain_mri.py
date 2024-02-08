import os
import random
from label_anything.data.test import LabelAnythingTestDataset
from label_anything.data.utils import BatchKeys
from PIL import Image
import json
import torch
from torchvision import transforms
from pycocotools import mask as mask_utils


class BrainMriTestDataset(LabelAnythingTestDataset):
    num_classes = 1
    def __init__(
        self,
        annotations: str,
        img_dir: str,
        preprocess=None,
    ):
        super().__init__()
        with open(annotations, "r") as f:
            annotations = json.load(f)
        self.annotations = annotations
        self.img_dir = img_dir  # data/raw/lgg-mri-segmentation/kaggle_3m/
        self.preprocess = preprocess
        self.image_to_category = self._image_to_category()

    def _image_to_category(self):
        image_to_category = {}
        for annotation in self.annotations["annotations"]:
            image_id = annotation["image_id"]
            category_id = annotation["category_id"]
            image_to_category[image_id] = category_id
        return image_to_category

    def __len__(self):
        return len(self.annotations["images"])

    def extract_prompts(self):
        cat_images = [
            img for img, cat_id in self.image_to_category.items() if cat_id == 1
        ]
        selected_images = random.sample(cat_images, min(5, len(cat_images)))
        image_data = [self._get_image_by_id(image_id) for image_id in selected_images]
        images, sizes = zip(*image_data)
        images = torch.stack(images)
        sizes = torch.stack(sizes)

        gt_data = [self._get_gt_by_id(image_id) for image_id in selected_images]
        masks, _ = zip(*gt_data)
        masks = torch.stack(masks)
        flag_masks = torch.tensor([1 if torch.sum(mask) > 0 else 0 for mask in masks])

        return {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.DIMS: sizes,
        }

    def _get_image_by_id(self, image_id):
        for image in self.annotations["images"]:
            if image["id"] == image_id:
                return self._get_image(image)

    def _get_gt_by_id(self, image_id):
        for annotation in self.annotations["annotations"]:
            if annotation["image_id"] == image_id:
                return self._get_gt(annotation)

    def _get_image(self, image_info):
        image_path = os.path.join(self.img_dir, image_info["url"])
        img = Image.open(image_path)
        size = img.size
        if self.preprocess:
            img = self.preprocess(img)  # 3 x h x w
        return img, torch.tensor(size)

    def _get_gt(self, annotation_info):
        mask = Image.fromarray(mask_utils.decode(annotation_info["segmentation"]))
        if self.preprocess:
            mask = self.preprocess(mask)
        return mask, torch.tensor(annotation_info["bbox"])

    def __getitem__(self, idx):
        image_info = self.annotations["images"][idx]
        annotation_info = self.annotations["annotations"][idx]
        image, _ = self._get_image(image_info)
        size = torch.tensor(image.shape[1:]).unsqueeze(0)  # Example dimension
        gt = self._get_gt(annotation_info)
        return {
            BatchKeys.IMAGES: image,
            BatchKeys.DIMS: size,
        }, gt


if __name__ == "__main__":
    from label_anything.data.transforms import CustomNormalize, CustomResize

    annotations = "/home/emanuele/LabelAnything/data/annotations/brain_mri.json"
    img_dir = "/home/emanuele/LabelAnything/data/raw/lgg-mri-segmentation/kaggle_3m/"
    preprocess = transforms.Compose(
        [
            CustomResize(1024),
            transforms.PILToTensor(),
            CustomNormalize(),
        ]
    )
    dataset = BrainMriTestDataset(annotations, img_dir, preprocess=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(dataset))
    print(next(iter(dataloader)))
