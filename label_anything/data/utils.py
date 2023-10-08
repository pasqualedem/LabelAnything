from PIL import Image, ImageDraw
import torch
import numpy as np
import itertools
import json


def compute_j_index(class_a, class_b):
    class_a = set(class_a)
    class_b = set(class_b)
    return len(class_a.intersection(class_b)) / len(class_a.union(class_b))


def convert_polygons(polygons):
    return [[(int(pol[i]), int(pol[i + 1])) for i in range(0, len(pol), 2)] for pol in polygons]


def get_mask(img_shape, reshape, segmentations):
    if segmentations == [[]]:  # for empty segmentation
        return reshape(torch.zeros(img_shape))[0]
    image = Image.new('L', img_shape[1:][::-1], 0)  # due to problem with shapes
    draw = ImageDraw.Draw(image)
    for pol in convert_polygons(segmentations):
        draw.polygon(pol, outline=1, fill=1)
    mask = np.asarray(image)
    mask = torch.Tensor(mask)
    return reshape(mask.unsqueeze(dim=0))


def get_mask_per_image(annotations, image_id, image_shape, reshape, target_classes):
    return torch.stack([
        get_mask(image_shape,
                 reshape,
                 itertools.chain(*annotations[(annotations.image_id == image_id) &
                                              (annotations.category_id == x)].segmentation.tolist()))
        for x in target_classes
    ])


def get_prompt_mask(annotations, image_shape, reshape, target_classes):
    return torch.stack([
        get_mask_per_image(annotations, x, image_shape, reshape, target_classes)
        for x in annotations.image_id.unique().tolist()
    ])


def get_bboxes(bboxes_entries, image_id, category_id, len_bbox):
    bbox = torch.Tensor(
        bboxes_entries[(bboxes_entries.image_id == image_id) & (bboxes_entries.category_id == category_id)]['bbox'].tolist()
    )
    ans = torch.cat([
        bbox, torch.zeros((len_bbox - bbox.size(0), 4))
    ], dim=0)
    if bbox.size(0) == 0:
        flag = torch.full((len_bbox, ), fill_value=False)
    elif bbox.size(0) == len_bbox:
        flag = torch.full((len_bbox, ), fill_value=True)
    else:
        flag = torch.cat(
            [torch.full((bbox.size(0), ), fill_value=True), torch.full(((len_bbox - bbox.size(0)), ), fill_value=False)]
        )
    return ans, flag


def get_prompt_bbox_per_image(bbox_entries, img_id, target_classes, max_anns):
    res = [get_bboxes(bbox_entries, img_id, x, max_anns) for x in target_classes]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def get_prompt_bbox(bbox_entries, target_classes):
    max_anns = get_max_bbox(bbox_entries)
    res = [get_prompt_bbox_per_image(bbox_entries, x, target_classes, max_anns)
           for x in bbox_entries.image_id.unique().tolist()]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def get_max_bbox(annotations):
    return max(
        len(annotations[(annotations.image_id == img) & (annotations.category_id == cat)])
        for img in annotations.image_id.unique() for cat in annotations.category_id.unique()
    )


def get_gt(annotations, image_shape, target_classes):
    gt = Image.new('L', image_shape[1:][::-1], 0)
    draw = ImageDraw.Draw(gt)
    for ix, c in enumerate(target_classes, start=1):
        polygons = convert_polygons(itertools.chain(*annotations[annotations.category_id == c].segmentation.tolist()))
        for pol in polygons:
            draw.polygon(pol, outline=1, fill=ix)
    gt = np.asarray(gt)
    gt = torch.Tensor(gt)
    return gt


def load_instances(json_path):
    with open(json_path) as f:
        instances = json.load(f)
    return instances