from typing import Any
from torchvision.transforms.functional import resize
from PIL import Image
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
import numpy as np
from copy import deepcopy
from typing import Tuple


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


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
        self, mean: Any = [123.675, 116.28, 103.53], std: Any = [58.395, 57.12, 57.375]
    ):
        self.pixel_mean = torch.tensor(mean).view(-1, 1, 1)
        self.pixel_std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: torch.Tensor):
        """
        Normalize the image.
        """
        sample = (sample - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = sample.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
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

    def convert_bbox(self, bbox):
        # convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = bbox
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return [x1, y1, x2, y2]

    def convert_mask(self, mask, h, w):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            ann (dict) : annotation object
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
