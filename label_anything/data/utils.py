import glob
import itertools
import json
import pickle
from enum import IntEnum, Enum
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class StrEnum(str, Enum):
    pass


class PromptType(StrEnum):
    BBOX = "bbox"
    MASK = "mask"
    POINT = "point"


class Label(IntEnum):
    POSITIVE = 1
    NULL = 0
    NEGATIVE = -1


class AnnFileKeys(StrEnum):
    IMAGES = "images"
    ANNOTATIONS = "annotations"
    CATEGORIES = "categories"
    ID = "id"
    IMAGE_ID = "image_id"
    CATEGORY_ID = "category_id"
    IMAGE = "image"
    ISCROWD = "iscrowd"


class BatchKeys(StrEnum):
    IMAGES = "images"
    EMBEDDINGS = "embeddings"
    PROMPT_MASKS = "prompt_masks"
    FLAG_MASKS = "flag_masks"
    PROMPT_POINTS = "prompt_points"
    FLAG_POINTS = "flag_points"
    PROMPT_BBOXES = "prompt_bboxes"
    FLAG_BBOXES = "flag_bboxes"
    DIMS = "dims"
    CLASSES = "classes"
    IMAGE_IDS = "image_ids"
    GROUND_TRUTHS = "ground_truths"
    
    
class BatchMetadataKeys(StrEnum):
    PROMPT_TYPES = "prompt_types"
    NUM_EXAMPLES = "num_examples"


def cast_type(input, dtype) -> dict:
    """Casts the values of the input dictionary / tuple / list / Tensor to specified dtype if different from fp32."""
    if dtype == torch.float:
        return input
    if isinstance(input, dict):
        return {key: cast_type(value, dtype) for key, value in input.items()}
    if isinstance(input, tuple):
        return tuple(cast_type(value, dtype) for value in input)
    if isinstance(input, list):
        return [cast_type(value, dtype) for value in input]
    if isinstance(input, torch.Tensor) and input.dtype == torch.float32:
        return input.type(dtype)
    return input


def get_max_annotations(annotations: list) -> int:
    """Returns the maximum number of annotations for a single image.

    Args:
        annotations (list): list of annotations.

    Returns:
        int: maximum number of annotations for a single image.
    """
    anns = []
    for image in annotations:
        for cat in image:
            anns.append(image[cat].shape[0])
    return max(anns)


def load_dict(path: str) -> dict:
    """Loads a dictionary from a file.

    Args:
        path (str): path to the file containing the dictionary.

    Returns:
        dict: dictionary.
    """
    _, ext = str(path).split(".")
    if ext == "json":
        with open(path, "r") as f:
            instances = json.load(f)
    elif ext in {"pickle", "pkl"}:
        print("Using pickle")
        with open(path, "rb") as f:
            instances = pickle.load(f)
    else:
        raise ValueError("Invalid file extension")
    return instances


def load_instances(path: str) -> dict:
    """Loads a dictionary of instances from a file.

    Args:
        path (str): path to the file containing the instances.

    Returns:
        dict: dictionary of instances.
    """
    if "*" in str(path):
        files = glob.glob(path)
        instances = {}
        for file in files:
            instances.update(load_dict(file))
    else:
        instances = load_dict(path)
    return instances


# COLLATE UTILS


def rearrange_classes(classes: List[Dict[int, int]]) -> Dict[int, int]:
    """
    Returns a new dict for class positions in a batch
    """
    distinct_classes = set(itertools.chain(*[list(x.values()) for x in classes]))
    return {val: ix for ix, val in enumerate(distinct_classes, start=1)}


def collate_gt(
    tensor: torch.Tensor, original_classes: Dict[int, int], new_classes: Dict[int, int]
) -> torch.Tensor:
    """
    Rearranges the ground truth mask for a single query image, replacing the old value of the class with the new one.

    Arguments:
        tensor: original ground truth mask of shape H x W.
        original_classes: dict in which each pair k: v is ith class corresponding to class id.
        new_classes: dict in which each pair k: v is class id corresponding to new jth class.

    Returns:
        torch.Tensor: new ground truth tensor of shape H x W, in which the values are rearranged according with
        new classes dict values.
    """
    for i in range(tensor.size(0)):
        for j in range(tensor.size(0)):
            tensor[i, j] = (
                0
                if tensor[i, j].item() == 0
                else new_classes[original_classes[tensor[i, j].item()]]
            )
    return tensor


def collate_mask(
    masks: torch.Tensor,
    flags: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.tensor]:
    """
    Rearranges the mask tensor for a single query image, rearranging the shape, according to the classes present in
    the whole batch.

    Arguments:
        tensor: mask tensor of shape M x C_old x H x W, including the masks of all examples of a single query image
                for all the classes referring to it.
        original_classes: dict in which each pair k: v is ith class corresponding to class id.
        new_classes: dict in which each pair k: v is class id corresponding to new jth class.

    Returns:
         torch.Tensor: new mask tensor of shape M x C_new x H x W, including the masks of all examples of a single query
                       image. The ith mask is rearranged such that it will be in jth position, according to the new
                       index in new classes' dict.
    """
    (
        m,
        c,
        h,
        w,
    ) = masks.shape
    out = torch.zeros(size=(m, num_classes, h, w), dtype=masks.dtype)
    out_flags = torch.zeros(size=(m, num_classes), dtype=flags.dtype)
    out[:, :c, :, :] = masks
    out_flags[:, :c] = flags

    return out, out_flags


def collate_bbox(
    bbox: torch.Tensor,
    flag: torch.Tensor,
    num_classes: int,
    max_annotations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rearranges the bbox tensor and its associated flag tensor for a single query image, according to the classes present
    in the whole batch.

    Arguments:
        bbox: tensor of shape M x C_old x N_old x 4, representing all the bounding boxes associated to the given example
              and the target classes required.
        flag: padding tensor of shape M x C_old x N_old, in which each element indicates whether the bounding box is
              real or a padding one.
        original_classes: dict in which each pair k: v is ith class corresponding to class id.
        new_classes: dict in which each pair k: v is class id corresponding to new jth class.
        max_annotations: maximum number of annotations present in the whole batch.

    Returns:
        torch.Tensor: new bounding box tensor of shape M x C_new x N_new x 4, with C_new equal to the number of elements
                      in new_classes dict and N_new equal to max annotations.
        torch.Tensor: new bounding box flag tensor of shape M x C_new x N_new, with C_new equal to the number of
                      elements in new_classes dict and N_new equal to max annotations.
    """
    m, c, n, b_dim = bbox.shape
    out_bbox = torch.zeros(size=(m, num_classes, max_annotations, b_dim))
    out_flag = torch.zeros(size=(m, num_classes, max_annotations))
    out_bbox[:, :c, :n, :] = bbox
    out_flag[:, :c, :n] = flag
    return out_bbox, out_flag


def collate_coords(
    coords: torch.Tensor,
    flag: torch.Tensor,
    num_classes: int,
    max_annotations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rearranges the coordinates tensor and its associated flag tensor for a single query image, according to the classes
    present in the whole batch.

    Arguments:
        coords: tensor of shape M x C_old x N_old x K x 2, representing the coordinates extracted for a single query
                image.
        flag: tensor of shape M x C_old x N_old x K, in which each element indicates whether the generated coordinate
              is real or a pad one.
        original_classes: dict in which each pair k: v is ith class corresponding to class id.
        new_classes: dict in which each pair k: v is class id corresponding to new jth class.
        max_annotations: maximum number of annotations present in the whole batch.

    Returns:
        torch.Tensor: rearranged prompt coords tensor of shape M x C_new x N_new x K x 2, with C_new equal to the number
                      of elements in new_classes dict and N_new equal to max annotations.
        torch.Tensor: new prompt coords flag tensor of shape M x C_new x N_new x K, with C_new equal to the number of
                      elements in new_classes dict and N_new equal to max annotations.
    """
    m, c, n, c_dim = coords.shape
    out_coords = torch.zeros(size=(m, num_classes, max_annotations, c_dim))
    out_flag = torch.zeros(size=(m, num_classes, max_annotations))
    out_coords[:, :c, :n, :] = coords
    out_flag[:, :c, :n] = flag
    return out_coords, out_flag


def collate_gts(gt, dims):
    """Collate ground truths for a single sample (query + support)."""
    out = torch.zeros(dims)
    dim0, dim1 = gt.size()
    out[:dim0, :dim1] = gt
    return out


def collate_batch_gts(gt, dims, fill_value=-100):
    """Collate ground truths for a batch of samples, here the fill_value must be -100."""
    out = torch.full(size=(gt.size(0), *dims), fill_value=fill_value, dtype=torch.long)
    _, dim0, dim1 = gt.shape
    out[:, :dim0, :dim1] = gt
    return out


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def random_item(num_examples, depth, height, width, num_classes, num_objects):
    return {
        "embeddings": torch.rand(num_examples, depth, height // 16, width // 16),
        "prompt_masks": torch.randint(
            0, 2, (num_examples, num_classes, height // 4, width // 4)
        ).float(),
        "flag_masks": torch.randint(0, 2, (num_examples, num_classes)),
        "prompt_points": torch.randint(
            0, 2, (num_examples, num_classes, num_objects, 2)
        ),
        "flag_points": torch.randint(0, 2, (num_examples, num_classes, num_objects)),
        "prompt_bboxes": torch.rand(num_examples, num_classes, num_objects, 4),
        "flag_bboxes": torch.randint(0, 2, (num_examples, num_classes, num_objects)),
        "dims": torch.tensor([height, width]),
        "classes": [{1, 2}, {2, 3}],
    }, torch.randint(0, 3, (num_examples, height // 4, width // 4))


def random_batch(
    batch_size, num_examples, depth, height, width, num_classes, num_objects
):
    return RandomDataset.collate_fn(
        None,
        [
            random_item(num_examples, depth, height, width, num_classes, num_objects)
            for _ in range(batch_size)
        ],
    )


class VariableDataset(Dataset):
    def __init__(self, examples_batch_pairs, num_classes, num_objects, height, width):
        self.examples_batch_pairs = examples_batch_pairs
        self.num_classes = num_classes
        self.num_objects = num_objects

    def __getitem__(self, index):
        (
            batch_size,
            num_examples,
        ) = self.examples_batch_pairs[index]
        return random_item(
            *self.examples_batch_pairs[index],
            self.num_classes,
            self.num_objects,
            self.height,
            self.width
        )


class RandomDataset(Dataset):
    def __init__(self):
        self.len = 100

    def __getitem__(self, index):
        H, W = 1024, 1024
        M = 2
        C = 3
        D = 256
        N = 5
        return {
            "embeddings": torch.rand(M, D, H // 16, W // 16),
            "prompt_masks": torch.randint(0, 2, (M, C, H // 4, W // 4)).float(),
            "flags_masks": torch.randint(0, 2, (M, C)),
            "prompt_points": torch.randint(0, 2, (M, C, N, 2)),
            "flags_points": torch.randint(0, 2, (M, C, N)),
            "prompt_bboxes": torch.rand(M, C, N, 4),
            "flags_bboxes": torch.randint(0, 2, (M, C, N)),
            "dims": torch.tensor([H, W]),
            "classes": [{1, 2}, {2, 3}],
        }, torch.randint(0, 3, (M, H // 4, W // 4))

    def __len__(self):
        return self.len

    def collate_fn(self, batch):
        result_dict = {}
        gt_list = []
        for elem in batch:
            dictionary, gts = elem
            gt_list.append(gts)
            for key, value in dictionary.items():
                if key in result_dict:
                    if not isinstance(result_dict[key], list):
                        result_dict[key] = [result_dict[key]]
                    if isinstance(value, list):
                        result_dict[key].extend(value)
                    else:
                        result_dict[key].append(value)
                else:
                    if isinstance(value, list):
                        result_dict[key] = value
                    else:
                        result_dict[key] = [value]
        return {
            k: torch.stack(v) if k != "classes" else v for k, v in result_dict.items()
        }, torch.stack(gt_list)
