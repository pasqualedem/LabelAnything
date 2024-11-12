import glob
import itertools
import json
import pickle
from enum import IntEnum, Enum
from itertools import combinations
from typing import Dict, List, Tuple


import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

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
    SEGMENTATION = "segmentation"


class BatchKeys(StrEnum):
    IMAGES = "images"
    EMBEDDINGS = "embeddings"
    PROMPT_MASKS = "prompt_masks"
    FLAG_MASKS = "flag_masks"
    PROMPT_POINTS = "prompt_points"
    FLAG_POINTS = "flag_points"
    PROMPT_BBOXES = "prompt_bboxes"
    FLAG_BBOXES = "flag_bboxes"
    FLAG_EXAMPLES = "flag_examples"
    DIMS = "dims"
    CLASSES = "classes"
    INTENDED_CLASSES = "intended_classes"
    IMAGE_IDS = "image_ids"
    GROUND_TRUTHS = "ground_truths"
    CLIP_EMBEDDINGS = "clip_embeddings"
    
    
class BatchMetadataKeys(StrEnum):
    PROMPT_TYPES = "prompt_types"
    NUM_EXAMPLES = "num_examples"
    NUM_CLASSES = "num_classes"
    PROMPT_CHOICE_LEVEL = "prompt_choice_level"
    
    
def flags_merge(flag_masks: torch.Tensor = None, flag_points: torch.Tensor = None, flag_bboxes: torch.Tensor = None) -> torch.Tensor:
    """
    Merges the flags of the prompt masks, points and bboxes into a single tensor.

    Args:
        flag_masks (torch.Tensor): tensor of shape M x C, in which each element indicates whether the prompt mask is real
                                   or a padding one.
        flag_points (torch.Tensor): tensor of shape M x C x N, in which each element indicates whether the prompt point is
                                    real or a padding one.
        flag_bboxes (torch.Tensor): tensor of shape M x C x N, in which each element indicates whether the prompt bbox is
                                    real or a padding one.

    Returns:
        torch.Tensor: tensor of shape M x C, in which each element indicates whether the example is real or a padding one.
    """
    if flag_masks is None and flag_points is None and flag_bboxes is None:
        raise ValueError("At least one of the flags must be provided.")
    flag_examples = []
    if flag_points is not None:
        flag_examples.append(flag_points.any(dim=-1))
    if flag_bboxes is not None:
        flag_examples.append(flag_bboxes.any(dim=-1))
    if flag_masks is not None:
        flag_examples.append(flag_masks)
    if len(flag_examples) > 1:
        flag_examples = torch.stack(flag_examples, dim=1).any(dim=1)
    else:
        flag_examples = flag_examples[0]
    
    # Put BG class to 1
    flag_examples[:, 0] = 1
    return flag_examples


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


def annotations_to_tensor(
    prompts_processor, annotations: list, img_sizes: list, prompt_type: PromptType
) -> torch.Tensor:
    """Convert a list of annotations to a tensor.

    Args:
        prompts_processor (PromptsProcessor): The prompts processor.
        annotations (list): A list of annotations.
        img_sizes (list): A list of tuples containing the image sizes.
        prompt_type (PromptType): The type of the prompt.

    Returns:
        torch.Tensor: The tensor containing the annotations.
    """
    n = len(annotations)
    c = len(annotations[0])

    if prompt_type == PromptType.BBOX:
        max_annotations = get_max_annotations(annotations)
        tensor_shape = (n, c, max_annotations, 4)
    elif prompt_type == PromptType.MASK:
        tensor_shape = (n, c, 256, 256)
    elif prompt_type == PromptType.POINT:
        max_annotations = get_max_annotations(annotations)
        tensor_shape = (n, c, max_annotations, 2)

    tensor = torch.zeros(tensor_shape)
    flag = (
        torch.zeros(tensor_shape[:-1]).type(torch.uint8)
        if prompt_type != PromptType.MASK
        else torch.zeros(tensor_shape[:2]).type(torch.uint8)
    )

    if prompt_type == PromptType.MASK:
        for i, annotation in enumerate(annotations):
            for j, cat_id in enumerate(annotation):
                mask = prompts_processor.apply_masks(annotation[cat_id])
                tensor_mask = torch.tensor(mask)
                tensor[i, j, :] = tensor_mask
                flag[i, j] = 1 if torch.sum(tensor_mask) > 0 else 0
    else:
        for i, (annotation, img_original_size) in enumerate(
            zip(annotations, img_sizes)
        ):
            for j, cat_id in enumerate(annotation):
                if annotation[cat_id].size == 0:
                    continue
                m = annotation[cat_id].shape[0]
                if prompt_type == PromptType.BBOX:
                    boxes_ann = prompts_processor.apply_boxes(
                        annotation[cat_id], img_original_size
                    )
                    tensor[i, j, :m, :] = torch.tensor(boxes_ann)
                elif prompt_type == PromptType.POINT:
                    points_ann = prompts_processor.apply_coords(
                        annotation[cat_id], img_original_size
                    )
                    tensor[i, j, :m, :] = torch.tensor(points_ann)
                flag[i, j, :m] = 1

    return tensor, flag


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


def collate_example_flags(example_flags: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Rearranges the flags tensor for a single query image, according to the classes present in the whole batch.

    Arguments:
        example_flags: tensor of shape M x C_old, in which each element indicates whether the example is real or a
                       padding one.
        num_classes: number of classes present in the whole batch.
    """
    m, c = example_flags.shape
    out = torch.zeros(size=(m, num_classes), dtype=example_flags.dtype)
    out[:, :c] = example_flags
    return out


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


def collate_flag_examples(flags, n_classes):
    n_ex, *_ = flags[0].size()
    out_flags = torch.zeros(n_ex * n_classes, n_classes, dtype=flags[0].dtype)
    for idx, f in enumerate(flags):
        out_flags[(idx*n_ex): ((idx + 1)*n_ex), idx] = f.squeeze(dim=1)
    return out_flags.unsqueeze(dim=0)

def collate_class_masks(masks, flags, n_classes):
    n_ex, _, h, w = masks[0].size()
    out_mask = torch.zeros(n_ex * n_classes, n_classes, h, w, dtype=masks[0].dtype)
    out_flags = torch.zeros(n_ex * n_classes, n_classes, dtype=flags[0].dtype)
    for idx, (m, f) in enumerate(zip(masks, flags)):
        out_mask[(idx*n_ex): ((idx + 1)*n_ex), idx, :, :] = m.squeeze(dim=1)
        out_flags[(idx*n_ex): ((idx + 1)*n_ex), idx] = f.squeeze(dim=1)
    return out_mask.unsqueeze(dim=0), out_flags.unsqueeze(dim=0)


def collate_class_bbox(bboxes, flags, n_classes, max_annotations):
    n_ex = bboxes[0].size(0)
    out_bbox = torch.zeros(n_ex * n_classes, n_classes, max_annotations, 4, dtype=bboxes[0].dtype)
    out_flags = torch.zeros(n_ex * n_classes, n_classes, max_annotations, dtype=flags[0].dtype)
    for idx, (b, f) in enumerate(zip(bboxes, flags)):
        out_bbox[(idx*n_ex): ((idx + 1)*n_ex), idx, :b.size(2), :] = b.squeeze(dim=1)
        out_flags[(idx*n_ex): ((idx + 1)*n_ex), idx, :f.size(2)] = f.squeeze(dim=1)
    return out_bbox.unsqueeze(dim=0), out_flags.unsqueeze(dim=0)


def collate_class_points(points, flags, n_classes, max_annotations):
    n_ex = points[0].size(0)
    out_points = torch.zeros(n_ex * n_classes, n_classes, max_annotations, 2, dtype=points[0].dtype)
    out_flags = torch.zeros(n_ex * n_classes, n_classes, max_annotations, dtype=points[0].dtype)
    for idx, (p, f) in enumerate(zip(points, flags)):
        out_points[(idx*n_ex): ((idx + 1)*n_ex), idx, :p.size(2), :] = p.squeeze(dim=1)
        out_flags[(idx*n_ex): ((idx + 1)*n_ex), idx, :f.size(2)] = f.squeeze(dim=1)
    return out_points.unsqueeze(dim=0), out_flags.unsqueeze(dim=0)


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


def get_mean_std(mean, std):
    str_to_mean = {
        "default": IMAGENET_DEFAULT_MEAN,
        "standard": IMAGENET_STANDARD_MEAN,
    }
    str_to_std = {
        "default": IMAGENET_DEFAULT_STD,
        "standard": IMAGENET_STANDARD_STD,
    }
    if isinstance(mean, str):
        mean = str_to_mean[mean]
    if isinstance(std, str):
        std = str_to_std[std]
    return mean, std


def to_global_multiclass(
    classes: list[list[list[int]]], categories: dict[int, dict], *tensors: list[torch.Tensor], compact=True
) -> list[torch.Tensor]:
    """Convert the classes of an episode to the global classes.

    Args:
        classes (list[list[list[int]]]): The classes corresponding to batch, episode and query.
        categories (dict[int, dict]): The categories of the dataset.
        compact (bool, optional): Whether to compact the categories. Defaults to True.

    Returns:
        list[Tensor]: The updated tensors.
    """
    batch_size = len(classes)
    out_tensors = [tensor.clone() for tensor in tensors]
    cats_map = {k: i + 1 for i, k in enumerate(categories.keys())}
    for i in range(batch_size):
        # assign to longest_classes the longest list in classes[i]
        longest_classes = sorted(list(set(sum(classes[i], []))))
        for j, v in enumerate(longest_classes):
            for tensor in out_tensors:
                value = cats_map[v] if compact else v
                tensor[i] = torch.where(tensor[i] == j + 1, value, tensor[i])
    return out_tensors