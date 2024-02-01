import os
import torch
import torchvision

from label_anything.data.utils import BatchKeys, collate_gts
from label_anything.data.dataset import LabelAnythingTestDataset


class WeedMapDataset(LabelAnythingTestDataset):
    def __init__(
        self,
        train_root,
        test_root,
        channels,
        transform=None,
        target_transform=None,
        return_path=False,
        prompt_images=None,
    ):
        super().__init__(self)
        self.train_root = train_root
        self.test_root = test_root
        self.channels = channels
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        if prompt_images is None:
            prompt_images = [
                # List of selected images from the training set
            ]
        self.prompt_images = prompt_images

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
        return len(os.listdir(self.gt_folder))

    def extract_prompts(self):
        images = [
            self._get_image(self.train_channels_folder, filename)
            for filename in self.prompt_images
        ]
        images = torch.stack(images)
        masks = [
            self._get_gt(self.train_gt_folder, filename)
            for filename in self.prompt_images
        ]
        masks = torch.stack(masks)
        contains_crop = (masks == 1).sum(dim=(1, 2)) > 0
        contains_weed = (masks == 2).sum(dim=(1, 2)) > 0
        flag_masks = torch.stack([contains_crop, contains_weed]).T 
        
        sizes = torch.stack([x.shape[1:] for x in images])
        
        prompt_dict = {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict
    
    def _get_image(self, channels_folder, filename):
        channels = []
        for channel_folder in channels_folder:
            channel_path = os.path.join(channel_folder, filename)
            channel = torchvision.io.read_image(channel_path)
            channel = self.transform(channel)
            channels.append(channel)
        channels = torch.cat(channels).float()
        return channels
    
    def _get_gt(self, gt_folder, img_filename):
        gt_filename = f"003_{img_filename.split('.')[0]}_GroundTruth_iMap.png"
        path = os.path.join(gt_folder, gt_filename)
        gt = torchvision.io.read_image(path)
        gt = gt[[2, 1, 0], ::]
        gt = gt.argmax(dim=0)
        gt = self.target_transform(gt)
        return gt

    def __getitem__(self, i):
        filename = self.test_images[i]
        gt = self._get_gt(i, self.train_gt_folder, filename)
        image = self._get_image(i)
        
        return {
            BatchKeys.IMAGES: image,
            BatchKeys.DIMS: image.shape[1:],
            BatchKeys.GROUND_TRUTHS: gt,
        }
