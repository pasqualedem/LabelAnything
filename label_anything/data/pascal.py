import json
import os
import random
from pycocotools import mask as mask_utils
from PIL import Image
from torch.nn.functional import one_hot
import torch

from label_anything.data.utils import BatchKeys, PromptType
from label_anything.data.test import LabelAnythingTestDataset


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

if __name__ == "__main__":
    from label_anything.data.transforms import CustomNormalize, CustomResize
    from torchvision import transforms

    instances_path = "/home/emanuele/LabelAnything/data/annotations/instances_voc12.json"
    img_dir = "/home/emanuele/LabelAnything/data/raw/VOCdevkit/VOC2012"
    preprocess = transforms.Compose(
        [
            CustomResize(1024),
            transforms.PILToTensor(),
            CustomNormalize(),
        ]
    )
    dataset = PascalVOCTestDataset(instances_path, img_dir, preprocess=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(dataset))
    print(next(iter(dataloader)))