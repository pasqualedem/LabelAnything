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


def preprocess_to_batch(query_image, support_set, classes):
    classes = [-1] + classes
    prompts_processor = PromptsProcessor()
    size = 1024
    transforms = Compose([CustomResize(size), PILToTensor(), CustomNormalize(size)])

    if not support_set:
        return {}

    images = [(elem["img"]) for elem in support_set.values()]
    image_sizes = [(img.size[1], img.size[0]) for img in images]

    bboxes = [{cat_id: [] for cat_id in classes} for _ in images]
    masks = [{cat_id: [] for cat_id in classes} for _ in images]
    points = [{cat_id: [] for cat_id in classes} for _ in images]
    for image_id, (elem, image_size) in enumerate(zip(support_set.values(), image_sizes)):
        for bbox, label in zip(elem["prompts"]["bboxes"], elem["prompts"]["labels"]):
            bboxes[image_id][label] = prompts_processor.convert_bbox(
                bbox,
                *image_size,
            )

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
    images = torch.stack([transforms(query_image)] + [transforms(img) for img in images])

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
