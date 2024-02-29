import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from einops import rearrange
from label_anything.data.transforms import CustomNormalize, CustomResize, Denormalize

from label_anything.data.utils import BatchKeys, get_preprocess_shape


class SAMFewShotModel(nn.Module):
    def __init__(self, sam, fewshot, image_size=1024):
        super(SAMFewShotModel, self).__init__()
        self.sam = sam
        self.fewshot = fewshot
        self.sam_image_size = image_size

        self.predict = None
        self.generate_class_embeddings = None

    def _depad_images(self, images, original_sizes):
        original_sizes = rearrange(original_sizes, "b m xy -> (b m) xy")
        input_sizes = [
            get_preprocess_shape(h, w, self.sam_image_size) for (h, w) in original_sizes
        ]  # these are the input sizes without padding

        # interpolate masks to the model size
        images = F.interpolate(
            images,
            (self.sam_image_size, self.sam_image_size),
            mode="bilinear",
            align_corners=False,
        )
        # remove padding from masks
        images = [
            images[i, :, : input_size[0], : input_size[1]]
            for i, input_size in enumerate(input_sizes)
        ]
        # interpolate masks to the original size
        images = [
            F.interpolate(
                torch.unsqueeze(images[i], 0),
                original_size.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for i, original_size in enumerate(original_sizes)
        ]
        return images

    def forward(self, input_dict):
        B, M, C = input_dict[BatchKeys.FLAG_EXAMPLES].shape
        # Remove query image from image set
        images = input_dict[BatchKeys.IMAGES]
        input_dict[BatchKeys.IMAGES] = images[:, 1:]
        sam_masks = self.sam(input_dict)

        if sam_masks is not None:
            input_dict[BatchKeys.PROMPT_MASKS][:, :, 1:][
                input_dict[BatchKeys.FLAG_MASKS][:, :, 1:].logical_not()
            ] = sam_masks.squeeze(1).float()

        swin_mean = torch.tensor(
            [0.485 * 255, 0.456 * 255, 0.406 * 255],
            device=input_dict[BatchKeys.IMAGES].device,
        )
        swin_std = torch.tensor(
            [0.229 * 255, 0.224 * 255, 0.225 * 255],
            device=input_dict[BatchKeys.IMAGES].device,
        )
        swin_size = 384

        preprocess = Compose(
            [
                Resize(size=(swin_size, swin_size)),
                Denormalize(device=input_dict[BatchKeys.IMAGES].device),
                Normalize(swin_mean, swin_std),
            ]
        )
        images = rearrange(images, "b m c h w -> (b m) c h w")
        images = self._depad_images(images, input_dict[BatchKeys.DIMS])
        images = torch.cat([preprocess(image) for image in images], dim=0)
        input_dict[BatchKeys.IMAGES] = rearrange(
            images, "(b m) c h w -> b m c h w", b=B, m=M + 1
        )
        return self.fewshot(input_dict)

    def get_learnable_params(self, train_params):
        return self.fewshot.parameters()
