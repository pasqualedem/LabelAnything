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

from label_anything.demo.utils import canvas_to_prompt_type, color_to_class


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
    

def extract_prompts_from_canvas(canvas, source_shape, target_shape):
    if "bboxes" in canvas: # already in the right format
        return canvas
    prompts = {"bboxes": [], "points": [], "masks": []}
    if "objects" not in canvas:
        return prompts
    objects = canvas["objects"]
    for obj in objects:
        if obj["type"] in ["circle", "rect", "path"]:
            prompt_type = canvas_to_prompt_type[obj["type"]]
            label = color_to_class(obj["fill"])
            if prompt_type == "bbox":
                bbox = [
                            obj["left"],
                            obj["top"],
                            obj["left"] + obj["width"],
                            obj["top"] + obj["height"],
                        ]
                bbox = reshape_bbox(bbox, source_shape, target_shape)
                prompts["bboxes"].append(
                    {
                        "bbox": bbox,
                        "label": label,
                    }
                )
            elif prompt_type == "point":
                point = [
                            obj["left"] + obj["radius"],
                            obj["top"] + obj["radius"],
                        ]
                point = reshape_point(point, source_shape, target_shape)
                prompts["points"].append(
                    {
                        "point": point,
                        "label": label,
                    }
                )
            elif prompt_type == "mask":
                mask = canvas_to_coco_path(obj["path"])
                mask = reshape_mask(mask, source_shape, target_shape)
                prompts["masks"].append(
                    {
                        "mask": mask,
                        "label": label,
                    }
                )
    return prompts


def preprocess_to_batch(query_image, support_set, classes, custom_preprocess=True):
    classes = [-1] + classes
    prompts_processor = PromptsProcessor(custom_preprocess=custom_preprocess)
    size = 1024
    transforms = Compose([CustomResize(size), PILToTensor(), CustomNormalize(size)])

    if not support_set:
        return {}

    images = [(elem.img) for elem in support_set.values()]
    image_sizes = [(img.size[1], img.size[0]) for img in images]
    
    for v, image_size in zip(support_set.values(), image_sizes):
        v.prompts = extract_prompts_from_canvas(v.prompts, v.reshape, image_size)

    bboxes = [{cat_id: [] for cat_id in classes} for _ in images]
    masks = [{cat_id: [] for cat_id in classes} for _ in images]
    points = [{cat_id: [] for cat_id in classes} for _ in images]

    for image_id, (elem, image_size) in enumerate(
        zip(support_set.values(), image_sizes)
    ):
        for bbox in elem.prompts["bboxes"]:
            label = bbox["label"]
            bbox = bbox["bbox"]
            bboxes[image_id][label].append(
                prompts_processor.convert_bbox(
                    bbox,
                    *image_size,
                )
            )

        for point in elem.prompts["points"]:
            label = point["label"]
            point = point["point"]
            points[image_id][label].append(point)
        for mask in elem.prompts["masks"]:
            label = mask["label"]
            mask = prompts_processor.convert_mask(mask["mask"], *image_size)
            masks[image_id][label].append(mask)

    for i in range(len(images)):
        for cat_id in classes:
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
    dims = torch.tensor([(query_image.size[1], query_image.size[0])] + image_sizes)
    images = torch.stack(
        [transforms(query_image)] + [transforms(img) for img in images]
    )

    data_dict = {
        utils.BatchKeys.IMAGES: images.unsqueeze(0).cuda(),
        utils.BatchKeys.PROMPT_MASKS: masks.unsqueeze(0).cuda(),
        utils.BatchKeys.FLAG_MASKS: flag_masks.unsqueeze(0).cuda(),
        utils.BatchKeys.PROMPT_POINTS: points.unsqueeze(0).cuda(),
        utils.BatchKeys.FLAG_POINTS: flag_points.unsqueeze(0).cuda(),
        utils.BatchKeys.PROMPT_BBOXES: bboxes.unsqueeze(0).cuda(),
        utils.BatchKeys.FLAG_BBOXES: flag_bboxes.unsqueeze(0).cuda(),
        utils.BatchKeys.FLAG_EXAMPLES: flag_examples.unsqueeze(0).cuda(),
        utils.BatchKeys.DIMS: dims.unsqueeze(0).cuda(),
        utils.BatchKeys.CLASSES: [classes[1:]],
    }

    return data_dict
