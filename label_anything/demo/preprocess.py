import numpy as np
import torch
import streamlit as st
from torchvision.transforms import Compose, PILToTensor, Resize, Normalize, ToTensor

from label_anything.data.transforms import (
    CustomNormalize,
    CustomResize,
    PromptsProcessor,
)
from label_anything.data import utils
from label_anything.demo.utils import canvas_to_prompt_type, color_to_class, debug_write


from torchvision.transforms.functional import resize
from PIL import Image

def canvas_to_coco_path(path):
    m, x, y = zip(*path[:-1])
    path = [val for pair in zip(x, y) for val in pair]
    return [path]


def reshape_bbox(bbox, source_shape, target_shape):
    bbox = np.array(bbox)
    bbox = bbox * np.array([target_shape[1] / source_shape[1], target_shape[0] / source_shape[0]] * 2)
    return bbox.tolist()


def reshape_point(point, source_shape, target_shape):
    point = np.array(point)
    point = point * np.array([target_shape[1] / source_shape[1], target_shape[0] / source_shape[0]])
    return point.tolist()


def reshape_mask(mask, source_shape, target_shape):
    mask = np.array(mask)
    mask = mask.reshape(1, 2, -1)
    reshaper = np.array([target_shape[1] / source_shape[1], target_shape[0] / source_shape[0]]).reshape(1, 2, 1)
    mask = mask * reshaper
    mask = mask.reshape(1, -1)
    return mask.tolist()

def extract_mask_from_image(canvas, source_shape, target_shape):
    """
    Extracts mask from image for each class using the color map and reshapes the mask to the target shape

    Args:
        canvas (dict): Containes the image and the color map
        source_shape (tuple): shape of the image
        target_shape (tuple): shape of the target image
    """
    mask = canvas["mask"]
    colormap = canvas["color_map"]
    masks = {cls: np.all(mask == color, axis=2) for cls, color in colormap.items()}
    # Background always empty
    masks[-1] = np.zeros(mask.shape[:2], dtype=bool)
    masks =  [
            {
                "mask": masks[i],
                "label": i,
            }
            for i in masks.keys()
        ]
    
    return {
        "masks": masks,
        "points": [],
        "bboxes": [],
    }
    

def extract_prompts_from_canvas(canvas, source_shape, target_shape):
    if "bboxes" in canvas: # already in the right format
        return canvas
    # Check if we uploaded a mask
    if "mask" in canvas and "color_map" in canvas:
        return extract_mask_from_image(canvas, source_shape, target_shape)
    prompts = {"bboxes": [], "points": [], "masks": []}
    if "objects" not in canvas:
        return prompts
    objects = canvas["objects"]
    for obj in objects:
        if obj["type"] == "path":
            label = color_to_class(obj["fill"])
            is_focused = obj["stroke"] == "white"
            mask = canvas_to_coco_path(obj["path"])
            mask = reshape_mask(mask, source_shape, target_shape)
            prompts["masks"].append(
                {
                    "mask": mask,
                    "label": label,
                    "focused": is_focused
                }
            )
    return prompts

def reduce_masks(masks: list, focusing_factor: int):
    masks_side_length = 256
    # take the binary OR over the masks and resize to new size

    if not masks:
        return torch.zeros(
            (masks_side_length, masks_side_length), dtype=torch.uint8
        )

    masks, focused = zip(*masks)
    masks = np.array(masks)

    mask = torch.as_tensor(np.logical_or.reduce(masks).astype(np.uint8)).unsqueeze(
        0
    )
    if any(focused):
        focused = np.array(focused)
        focused_masks = masks[focused]
        focused_mask = torch.as_tensor(np.logical_or.reduce(focused_masks).astype(np.uint8)).unsqueeze(
            0
        )
        mask[focused_mask.bool()] = focusing_factor

    mask = resize(
        mask,
        (masks_side_length, masks_side_length),
        interpolation=Image.NEAREST,
    )
    return mask

def annotation_mask_to_tensor(annotations: list, focusing_factor: int) -> torch.Tensor:
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

    tensor_shape = (n, c, 256, 256)

    tensor = torch.zeros(tensor_shape)
    flag = torch.zeros(tensor_shape[:2]).type(torch.uint8)
    for i, annotation in enumerate(annotations):
        for j, cat_id in enumerate(annotation):
            mask = reduce_masks(annotation[cat_id], focusing_factor)
            tensor_mask = torch.tensor(mask)
            tensor[i, j, :] = tensor_mask
            flag[i, j] = 1 if torch.sum(tensor_mask) > 0 else 0

    return tensor, flag


def preprocess_support_set(support_set, classes, preprocess, custom_preprocess=True, device="cuda", focusing_factor=5):
    classes = [-1] + classes
    prompts_processor = PromptsProcessor(custom_preprocess=custom_preprocess)
    
    if not support_set:
        return {}

    images = [(elem.img) for elem in support_set]
    image_sizes = [(img.size[1], img.size[0]) for img in images]

    for v, image_size in zip(support_set, image_sizes):
        v.prompts = extract_prompts_from_canvas(v.prompts, v.reshape, image_size)
        
    masks = [{cat_id: [] for cat_id in classes} for _ in images]

    for image_id, (elem, image_size) in enumerate(
        zip(support_set, image_sizes)
    ):
        for mask in elem.prompts["masks"]:
            label = mask["label"]
            is_focused = mask.get("focused", False)
            # Check if mask or RLE
            if isinstance(mask["mask"], list):   
                mask = prompts_processor.convert_mask(mask["mask"], *image_size)
            else:
                mask = mask["mask"]
            masks[image_id][label].append((mask, is_focused))
            
    masks, flag_masks = annotation_mask_to_tensor(masks, focusing_factor)
    
    flag_examples = utils.flags_merge(flag_masks=flag_masks)
    dims = torch.tensor(image_sizes)
    images = torch.stack([preprocess(img) for img in images])

    return {
        utils.BatchKeys.IMAGES: images.unsqueeze(0).to(device),
        utils.BatchKeys.PROMPT_MASKS: masks.unsqueeze(0).to(device),
        utils.BatchKeys.FLAG_MASKS: flag_masks.unsqueeze(0).to(device),
        utils.BatchKeys.FLAG_EXAMPLES: flag_examples.unsqueeze(0).to(device),
        utils.BatchKeys.DIMS: dims.unsqueeze(0).to(device),
        utils.BatchKeys.CLASSES: [classes[1:]],
    }

def preprocess_to_batch(query_image, batch, preprocess, device="cuda",):
    dims = batch[utils.BatchKeys.DIMS].clone()
    images = batch[utils.BatchKeys.IMAGES].clone()
    dims = torch.cat([torch.tensor([[[query_image.size[1], query_image.size[0]]]], device=device), dims], dim=1)
    images = torch.cat(
        [preprocess(query_image).unsqueeze(0).unsqueeze(0).to(device), images],
        dim=1,
    )
    batch[utils.BatchKeys.IMAGES] = images
    batch[utils.BatchKeys.DIMS] = dims
    return batch


def denormalize(image):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=image.device
    )
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=image.device
    )
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    return (image * std) + mean
    