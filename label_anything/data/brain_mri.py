import os
import random
from label_anything.data.test import LabelAnythingTestDataset
from label_anything.data.utils import BatchKeys
from PIL import Image
import json
import torch
from torchvision import transforms
from pycocotools import mask as mask_utils
from torch.nn.functional import one_hot


class BrainMriTestDataset(LabelAnythingTestDataset):
    num_classes = 2

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
        flag_examples = flag_masks.clone().bool()

        prompt_dict = {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict

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
        gt = mask_utils.decode(annotation_info["segmentation"])
        if self.preprocess:
            gt = torch.from_numpy(gt)
        return gt.long()

    def __getitem__(self, idx):
        image_info = self.annotations["images"][idx]
        annotation_info = self.annotations["annotations"][idx]
        image, size = self._get_image(image_info)
        gt = self._get_gt(annotation_info)
        return {
            BatchKeys.IMAGES: image.unsqueeze(0),
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