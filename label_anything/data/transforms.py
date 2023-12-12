from typing import Any
from torchvision.transforms.functional import resize
from PIL import Image
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
import numpy as np
from copy import deepcopy
from typing import Tuple
from label_anything.data.utils import get_preprocess_shape


class CustomResize(object):
    def __init__(self, long_side_length: int = 1024):
        self.long_side_length = long_side_length

    def __call__(self, sample: Image):
        """
        Resize the image to the target long side length.
        """
        oldw, oldh = sample.size
        target_size = get_preprocess_shape(oldh, oldw, self.long_side_length)
        return resize(sample, target_size)


class CustomNormalize(object):
    def __init__(
        self, long_side_length: int = 1024, mean: Any = [123.675, 116.28, 103.53], std: Any = [58.395, 57.12, 57.375]
    ):
        self.long_side_length = long_side_length
        self.pixel_mean = torch.tensor(mean).view(-1, 1, 1)
        self.pixel_std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: torch.Tensor):
        """
        Normalize the image.
        """
        sample = (sample - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = sample.shape[-2:]
        padh = self.long_side_length - h
        padw = self.long_side_length - w
        sample = F.pad(sample, (0, padw, 0, padh))
        return sample


class PromptsProcessor:
    def __init__(self, long_side_length: int = 1024, masks_side_length: int = 256):
        self.long_side_length = long_side_length
        self.masks_side_length = masks_side_length

    def __ann_to_rle(self, ann, h, w):
        """Convert annotation which can be polygons, to RLE.

        Args:
            ann (dict): annotation object
            h (int): image height
            w (int): image width
        """
        segm = ann
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann
        return rle
    
    def __add_bbox_noise(self, bbox, hb, wb, h, w):
        x1, y1, x2, y2 = bbox
        # take random number from normal distribution with mean 0 and std 0.1 * l
        noise_x1 = np.clip(np.random.normal(0, 0.1 * wb), -20, 20)
        noise_y1 = np.clip(np.random.normal(0, 0.1 * hb), -20, 20)
        noise_x2 = np.clip(np.random.normal(0, 0.1 * wb), -20, 20)
        noise_y2 = np.clip(np.random.normal(0, 0.1 * hb), -20, 20)

        x1 = float(np.clip(x1 + noise_x1, 0, w))
        y1 = float(np.clip(y1 + noise_y1, 0, h))
        x2 = float(np.clip(x2 + noise_x2, 0, w))
        y2 = float(np.clip(y2 + noise_y2, 0, h))

        return [x1, y1, x2, y2]


    def convert_bbox(self, bbox, h, w, noise=False):
        # convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        x, y, wb, hb = bbox
        x1 = x
        y1 = y
        x2 = x + wb
        y2 = y + hb
        if noise:
            return self.__add_bbox_noise([x1, y1, x2, y2], hb, wb, h, w)
        return [x1, y1, x2, y2]

    def convert_mask(self, mask, h, w):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            mask: mask can be polygons, uncompressed RLE, or RLE
            h (int): image height
            w (int): image width

        Returns:
            binary mask (numpy 2D array)
        """
        rle = self.__ann_to_rle(mask, h, w)
        matrix = mask_utils.decode(rle)
        unique_values = np.unique(matrix).tolist()
        if len(unique_values) == 1 and unique_values[0] == 0:
            pol = torch.as_tensor(mask).view(-1, 2)
            y, x = torch.mean(pol, 0).type(torch.int).tolist()
            matrix[x, y] = 1
        return matrix

    def sample_point(self, mask: np.ndarray):
        # make a list of positive (row, col) coordinates
        positive_coords = np.argwhere(mask)
        # choose one at random
        row, col = positive_coords[np.random.choice(len(positive_coords))]
        return col, row

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = get_preprocess_shape(original_size[0], original_size[1], 1024)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    def torch_apply_coords(
        self, coords: torch.tensor, original_size: Tuple[int, ...]
    ) -> torch.tensor:
        """
        Expects a torch of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = get_preprocess_shape(original_size[0], original_size[1], 1024)
        coords = coords.clone().float()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_masks(self, masks: np.ndarray):
        # take the binary OR over the masks and resize to new size
        if len(masks) == 0:
            return torch.zeros(
                (self.masks_side_length, self.masks_side_length), dtype=torch.uint8
            )

        mask = torch.as_tensor(np.logical_or.reduce(masks).astype(np.uint8)).unsqueeze(
            0
        )
        new_h, new_w = get_preprocess_shape(masks[0].shape[0], masks[0].shape[1], 1024)
        mask = resize(mask, (new_h, new_w), interpolation=Image.NEAREST)
        padw = 1024 - new_w
        padh = 1024 - new_h
        mask = F.pad(mask, (0, padw, 0, padh))
        mask = resize(
            mask,
            (self.masks_side_length, self.masks_side_length),
            interpolation=Image.NEAREST,
        )
        return mask
