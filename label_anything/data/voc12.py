import pathlib
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import PILToTensor

from label_anything.data.transforms import CustomNormalize, CustomResize


class VOC12Dataset(Dataset):
    def __init__(self, root, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.root = pathlib.Path(root)
        self.images = self.data["images"]
        self.annotations = self.data["annotations"]
        self.categories = self.data["categories"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]

        # Load the image
        image_path = self.root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        # Find all annotations for the current image
        image_annotations = [
            ann for ann in self.annotations if ann["image_id"] == image_id
        ]

        # For simplicity, let's assume each image has exactly one annotation
        # You might need to handle the case where an image has multiple annotations
        annotation = image_annotations[0]

        # Extract the bounding box and label
        bbox = annotation["bbox"]
        label = annotation["category_id"]

        # For the mask, let's assume it's stored in a separate file with the same name as the image
        mask_path = self.root / (image_info["file_name"] + "_mask.png")
        mask = Image.open(mask_path)

        return image, bbox, mask, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose

    preprocess = Compose(
        [
            CustomResize(1024),
            PILToTensor(),
            CustomNormalize(),
        ]
    )

    dataset = VOC12Dataset(
        dataset_path="/home/emanuele/Dottorato/dataset-vari/VOC12",
        instances_path="notebooks/instances_voc12.json",
        max_num_examples=10,
        preprocess=preprocess,
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    image, boxes, mask, labels = next(iter(dataloader))
    
    # print the shape of the image and the boxes
    print(f"image: {image.shape}")
    print(f"boxes: {boxes.shape}")
    print(f"labels: {labels.shape}")
    print(f"mask: {mask.shape}")
    # print(f"masks: {masks.shape}")
