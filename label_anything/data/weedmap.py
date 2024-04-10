import os
import torch
import torchvision

from PIL import Image
from torch.nn.functional import one_hot

from label_anything.data.utils import BatchKeys
from label_anything.data.test import LabelAnythingTestDataset


class WeedMapTestDataset(LabelAnythingTestDataset):
    id2class = {0: "background", 1: "crop", 2: "weed"}
    num_classes = 3
    def __init__(
        self,
        train_root,
        test_root,
        preprocess=None,
        prompt_images=None,
    ):
        super().__init__()
        self.train_root = train_root
        self.test_root = test_root
        self.transform = preprocess
        if prompt_images is None:
            prompt_images = [
                # List of selected images from the training set
                "frame0009_2.png",
                "frame0021_2.png",
                "frame0033_3.png",
                "frame0034_1.png",
                "frame0048_0.png",
            ]
        self.prompt_images = prompt_images
        self.channels = ["R", "G", "B"]

        self.train_gt_folder = os.path.join(self.train_root, "groundtruth")
        self.test_gt_folder = os.path.join(self.test_root, "groundtruth")
        self.test_channels_folder = [
            os.path.join(self.test_root, "tile", channel) for channel in self.channels
        ]
        self.train_channels_folder = [
            os.path.join(self.train_root, "tile", channel) for channel in self.channels
        ]
        self.train_images = os.listdir(self.train_channels_folder[0])
        self.test_images = os.listdir(self.test_channels_folder[0])
        

    def __len__(self):
        return len(os.listdir(self.test_gt_folder))
    
    def _transform(self, image):
        image = Image.fromarray(image.permute(1, 2, 0).numpy().astype("uint8"))
        image = self.transform(image)
        return image
        

    def extract_prompts(self):
        images = [
            self._get_image(self.train_channels_folder, filename)
            for filename in self.prompt_images
        ]
        sizes = torch.stack([torch.tensor(x.shape[1:]) for x in images])
        images = [
            self._transform(image)
            for image in images
        ]
        images = torch.stack(images)
        masks = [
            self._get_gt(self.train_gt_folder, filename)
            for filename in self.prompt_images
        ]
        masks = torch.stack(masks)
        # Background flags are always 0
        backflag = torch.zeros(masks.shape[0])
        contains_crop = (masks == 1).sum(dim=(1, 2)) > 0
        contains_weed = (masks == 2).sum(dim=(1, 2)) > 0
        flag_masks = torch.stack([backflag, contains_crop, contains_weed]).T
        
        masks = one_hot(masks.long(), 3).permute(0, 3, 1, 2).float() 

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
            BatchKeys.FLAG_EXAMPLES: flag_masks,
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict
    
    def _get_image(self, channels_folder, filename):
        channels = []
        for channel_folder in channels_folder:
            channel_path = os.path.join(channel_folder, filename)
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()
        return channels
    
    def _get_gt(self, gt_folder, img_filename):
        field_id = gt_folder.split("/")[-2]
        gt_filename = f"{field_id}_{img_filename.split('.')[0]}_GroundTruth_iMap.png"
        path = os.path.join(gt_folder, gt_filename)
        gt = Image.open(path)
        gt = torchvision.transforms.PILToTensor()(gt)[0]
        # Convert crop value 10000 to 1
        gt[gt == 10000] = 1
        return gt.long()

    def __getitem__(self, i):
        filename = self.test_images[i]
        gt = self._get_gt(self.test_gt_folder, filename)
        image = self._get_image(self.test_channels_folder, filename)
        size = torch.tensor(image.shape[1:]) # Example dimension
        image = self._transform(image)
        return {
            BatchKeys.IMAGES: image.unsqueeze(0),
            BatchKeys.DIMS: size,
        }, gt
