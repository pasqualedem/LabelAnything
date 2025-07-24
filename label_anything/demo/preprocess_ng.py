import numpy as np
import torch

from torchvision.transforms import Compose, PILToTensor, Resize, Normalize, ToTensor
from PIL import Image

from label_anything.data.transforms import (
    CustomNormalize,
    CustomResize,
    PromptsProcessor,
)

from label_anything.data import utils
from label_anything.data.transforms import PromptsProcessor

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


def preprocess_support_set(support_set, classes, size=1024, custom_preprocess=True, device="cuda"):
    class_ids = [-1] + list(range(len(classes)))
    prompts_processor = PromptsProcessor(custom_preprocess=custom_preprocess)
    transforms = Compose([CustomResize(size), ToTensor(), CustomNormalize(size)])
    
    print(f"Preprocessing {support_set} support images with classes: {classes[1:]}")

    support_images = [Image.open(elem["image"]) for elem in support_set]
    image_sizes = [(img.size[1], img.size[0]) for img in support_images]

    bboxes = [{cat_id: [] for cat_id in class_ids} for _ in support_images]
    masks = [{cat_id: [] for cat_id in class_ids} for _ in support_images]
    points = [{cat_id: [] for cat_id in class_ids} for _ in support_images]

    for image_id, elem in enumerate(support_set):
        annotations = elem.get("annotations", {})
        for class_name, bboxes_ann in annotations.get("bboxes", {}).items():
            class_id = classes.index(class_name)
            for bbox in bboxes_ann:
                bboxes[image_id][class_id].append(bbox)

        for class_name, points_ann in annotations.get("points", {}).items():
            class_id = classes.index(class_name)
            for point in points_ann:
                points[image_id][class_id].append(point)

        for class_name, masks_ann in annotations.get("masks", {}).items():
            class_id = classes.index(class_name)
            for mask in masks_ann:
                if isinstance(mask, list):
                    print(f"Converting mask for class {mask}")
                    mask = prompts_processor.convert_mask([mask], *image_sizes[image_id])
                masks[image_id][class_id].append(mask)

    for i in range(len(support_images)):
        for cat_id in class_ids:
            bboxes[i][cat_id] = np.array((bboxes[i][cat_id]))
            masks[i][cat_id] = np.array((masks[i][cat_id]))
            points[i][cat_id] = np.array((points[i][cat_id]))

    bboxes, flag_bboxes = utils.annotations_to_tensor(
        prompts_processor, bboxes, image_sizes, utils.PromptType.BBOX
    )
    masks, flag_masks = utils.annotations_to_tensor(
        prompts_processor, masks, image_sizes, utils.PromptType.MASK
    )
    points, flag_points = utils.annotations_to_tensor(
        prompts_processor, points, image_sizes, utils.PromptType.POINT
    )

    flag_examples = utils.flags_merge(flag_masks, flag_points, flag_bboxes)
    dims = torch.tensor(image_sizes)
    for img in support_images:
        img = img.convert("RGB")
    support_images = torch.stack([transforms(img) for img in support_images])

    return {
        utils.BatchKeys.IMAGES: support_images.unsqueeze(0).to(device),
        utils.BatchKeys.PROMPT_MASKS: masks.unsqueeze(0).to(device),
        utils.BatchKeys.FLAG_MASKS: flag_masks.unsqueeze(0).to(device),
        utils.BatchKeys.PROMPT_POINTS: points.unsqueeze(0).to(device),
        utils.BatchKeys.FLAG_POINTS: flag_points.unsqueeze(0).to(device),
        utils.BatchKeys.PROMPT_BBOXES: bboxes.unsqueeze(0).to(device),
        utils.BatchKeys.FLAG_BBOXES: flag_bboxes.unsqueeze(0).to(device),
        utils.BatchKeys.FLAG_EXAMPLES: flag_examples.unsqueeze(0).to(device),
        utils.BatchKeys.DIMS: dims.unsqueeze(0).to(device),
        utils.BatchKeys.CLASSES: [class_ids[1:]],
    }