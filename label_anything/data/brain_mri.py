import os
from label_anything.data.test import LabelAnythingTestDataset
from label_anything.data.utils import BatchKeys
from PIL import Image
import json
import torch
from torchvision import transforms
from pycocotools import mask as mask_utils


class BrainMriTestDataset(LabelAnythingTestDataset):
    num_classes = 2

    def __init__(self, annotations, img_dir, transform=None):
        super().__init__()
        with open(annotations, "r") as f:
            annotations = json.load(f)
        self.annotations = annotations
        self.img_dir = img_dir  # data/raw/lgg-mri-segmentation/kaggle_3m/
        self.transform = transform

    def __len__(self):
        return len(self.annotations["images"])

    def _get_image(self, image_info):
        image_path = os.path.join(self.img_dir, image_info["url"])
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)  # 3 x h x w
        return img, torch.tensor(img.shape[1:]).unsqueeze(0)

    def _get_gt(self, annotation_info):
        mask = mask_utils.decode(annotation_info["segmentation"])  #
        bbox = (annotation_info["bbox"],)  # [x, y, w, h]
        if self.transform:
            mask = self.transform(mask)
            bbox = torch.tensor(bbox)
        return {"mask": mask, "bbox": bbox}

    def __getitem__(self, idx):
        image_info = self.annotations["images"][idx]
        annotation_info = self.annotations["annotations"][idx]
        image, size = self._get_image(image_info)
        gt = self._get_gt(annotation_info)
        return {
            BatchKeys.IMAGES: image,
            BatchKeys.DIMS: size,
        }, gt


if __name__ == "__main__":
    annotations = "/home/emanuele/LabelAnything/data/annotations/brain_mri.json"
    img_dir = "/home/emanuele/LabelAnything/data/raw/lgg-mri-segmentation/kaggle_3m/"
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = BrainMriTestDataset(annotations, img_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(dataset))
    print(next(iter(dataloader)))
