from PIL import Image, ImageDraw
from itertools import combinations

import torch
import numpy as np
import itertools
import json

MAX_PIXELS_BBOX_NOISE = 20


def compute_j_index(class_a, class_b):
    class_a = set(class_a)
    class_b = set(class_b)
    return len(class_a.intersection(class_b)) / len(class_a.union(class_b))


def compute_j_index_n_sets(sets):
    return len(set.intersection(*sets)) / len(set.union(*sets))


def mean_pairwise_j_index(sets):
    return sum(compute_j_index(a, b) for a, b in combinations(sets, 2)) / (len(sets) * (len(sets) - 1))


def convert_polygons(polygons):
    return [
        [(int(pol[i]), int(pol[i + 1])) for i in range(0, len(pol), 2)]
        for pol in polygons
    ]


def get_mask(img_shape, reshape, segmentations):
    if segmentations == [[]]:  # for empty segmentation
        return reshape(torch.zeros(img_shape))[0]
    image = Image.new("L", img_shape[1:][::-1], 0)  # due to problem with shapes
    draw = ImageDraw.Draw(image)
    for pol in convert_polygons(segmentations):
        draw.polygon(pol, outline=1, fill=1)
    mask = np.asarray(image)
    mask = torch.Tensor(mask)
    return reshape(mask.unsqueeze(dim=0))


def get_mask_per_image(annotations, image_id, image_shape, reshape, target_classes):
    return torch.stack(
        [
            get_mask(
                image_shape,
                reshape,
                itertools.chain(
                    *annotations[
                        (annotations.image_id == image_id)
                        & (annotations.category_id == x)
                    ].segmentation.tolist()
                ),
            )
            for x in target_classes
        ]
    )


def get_prompt_mask(annotations, image_shape, reshape, target_classes):
    return torch.stack(
        [
            get_mask_per_image(annotations, x, image_shape, reshape, target_classes)
            for x in annotations.image_id.unique().tolist()
        ]
    )


def add_noise(bbox):
    x, y, x1, y1 = bbox
    std_w = abs(x - x1) / 10
    std_h = abs(y - y1) / 10
    noise_x = list(
        map(
            lambda val: val if abs(val) <= 20 else np.sign(val) * MAX_PIXELS_BBOX_NOISE,
            np.random.normal(loc=0, scale=std_w, size=2).tolist(),
        )
    )
    noise_y = list(
        map(
            lambda val: val if abs(val) <= 20 else np.sign(val) * MAX_PIXELS_BBOX_NOISE,
            np.random.normal(loc=0, scale=std_h, size=2).tolist(),
        )
    )
    return x + noise_x[0], y + noise_y[0], x1 + noise_x[1], y1 + noise_y[1]


def get_bboxes(bboxes_entries, image_id, category_id, len_bbox):
    bbox = bboxes_entries[
        (bboxes_entries.image_id == image_id)
        & (bboxes_entries.category_id == category_id)
    ]["bbox"].tolist()
    bbox = list(map(lambda x: add_noise(x), bbox))
    bbox = torch.Tensor(bbox)
    ans = torch.cat([bbox, torch.zeros((len_bbox - bbox.size(0), 4))], dim=0)
    if bbox.size(0) == 0:
        flag = torch.full((len_bbox,), fill_value=False)
    elif bbox.size(0) == len_bbox:
        flag = torch.full((len_bbox,), fill_value=True)
    else:
        flag = torch.cat(
            [
                torch.full((bbox.size(0),), fill_value=True),
                torch.full(((len_bbox - bbox.size(0)),), fill_value=False),
            ]
        )
    return ans, flag


def get_prompt_bbox_per_image(bbox_entries, img_id, target_classes, max_anns):
    res = [get_bboxes(bbox_entries, img_id, x, max_anns) for x in target_classes]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def get_prompt_bbox(bbox_entries, target_classes):
    max_anns = get_max_bbox(bbox_entries)
    res = [
        get_prompt_bbox_per_image(bbox_entries, x, target_classes, max_anns)
        for x in bbox_entries.image_id.unique().tolist()
    ]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def get_max_bbox(annotations):
    return max(
        len(
            annotations[
                (annotations.image_id == img) & (annotations.category_id == cat)
            ]
        )
        for img in annotations.image_id.unique()
        for cat in annotations.category_id.unique()
    )


def get_gt(annotations, image_shape, target_classes):
    gt = Image.new("L", image_shape[1:][::-1], 0)
    draw = ImageDraw.Draw(gt)
    for ix, c in enumerate(target_classes, start=1):
        polygons = convert_polygons(
            itertools.chain(
                *annotations[annotations.category_id == c].segmentation.tolist()
            )
        )
        for pol in polygons:
            draw.polygon(pol, outline=1, fill=ix)
    gt = np.asarray(gt)
    gt = torch.Tensor(gt)
    return gt


def load_instances(json_path):
    with open(json_path, "r") as f:
        instances = json.load(f)
    return instances


def get_coords(annotation, max_num_coords, original_shape, reshape):
    if annotation["segmentation"] == [[]]:  # empty segmentation
        return torch.Tensor([0.0, 0.0]), torch.Tensor([False])
    mask = get_mask(
        img_shape=original_shape,
        reshape=reshape,
        segmentations=annotation["segmentation"],
    ).squeeze(dim=0)
    # num_coords = np.random.randint(1, max_num_coords)
    num_coords = max_num_coords
    coords = torch.nonzero(mask == 1)
    if coords.size(0) < num_coords:  # for very small masks
        return torch.cat(
            [
                coords,
                torch.zeros(size=((num_coords - coords.size(0)), *coords.shape[1:])),
            ]
        ), torch.cat(
            [
                torch.full(size=(coords.size(0),), fill_value=True),
                torch.full(size=((num_coords - coords.size(0)),), fill_value=False),
            ]
        )
    indexes = np.random.randint(0, coords.size(0), size=num_coords).tolist()
    return coords[indexes], torch.full(size=(num_coords,), fill_value=True)


# forse non serve paddare qui
def pad_coords_image_class(num_coords, coords, flags):
    if coords.size(0) == num_coords:
        return coords, flags
    return torch.cat(
        [coords, torch.zeros((num_coords - coords.size(0)), 2)]
    ), torch.cat(
        [flags, torch.full(size=((num_coords - coords.size(0)),), fill_value=False)]
    )


def get_coords_per_image_class(
    annotations, image_id, class_id, num_coords, original_shape, reshape
):
    target_annotations = annotations[
        (annotations.image_id == image_id) & (annotations.category_id == class_id)
    ]
    if len(target_annotations) == 0:
        return torch.zeros(size=(1, num_coords, 2)), torch.full(
            size=(1, num_coords), fill_value=False
        )  # aggiungi una coordinata in testa per i punti
    coords_flags = [
        get_coords(row, num_coords, original_shape, reshape)
        for _, row in target_annotations.iterrows()
    ]
    coords = [x[0] for x in coords_flags]
    flags = [x[1] for x in coords_flags]
    return torch.stack(coords), torch.stack(flags)


def pad_coords_image(coords_flags, max_annotations):
    coords, flags = coords_flags
    if coords.size(0) == max_annotations:
        return coords, flags
    return torch.cat(
        [
            coords,
            torch.zeros(size=((max_annotations - coords.size(0)), *coords.shape[1:])),
        ]
    ), torch.cat(
        [
            flags,
            torch.full(
                size=((max_annotations - flags.size(0)), flags.size(1)),
                fill_value=False,
            ),
        ]
    )


def get_coords_per_image(
    annotations, image_id, target_classes, num_coords, original_shape, resize
):
    coords_flags = [
        get_coords_per_image_class(
            annotations, image_id, class_id, num_coords, original_shape, resize
        )
        for class_id in target_classes
    ]
    max_annotations = max([x[0].shape[0] for x in coords_flags])
    padded = [pad_coords_image(x, max_annotations) for x in coords_flags]
    coords = [x[0] for x in padded]
    flags = [x[1] for x in padded]
    return torch.stack(coords), torch.stack(flags)


def pad_coords(coords_flags, max_annotations):
    coords, flags = coords_flags
    if coords.size(1) == max_annotations:
        return coords, flags
    coords = coords.permute(1, 0, 2, 3)
    flags = flags.permute(1, 0, 2)
    return torch.cat(
        [
            coords,
            torch.zeros(size=((max_annotations - coords.size(0)), *coords.shape[1:])),
        ]
    ).permute(1, 0, 2, 3), torch.cat(
        [
            flags,
            torch.full(
                size=((max_annotations - flags.size(0)), *flags.shape[1:]),
                fill_value=False,
            ),
        ]
    ).permute(
        1, 0, 2
    )


def get_prompt_coords(annotations, target_classes, num_coords, original_shape, resize):
    coords_flags = [
        get_coords_per_image(
            annotations, image_id, target_classes, num_coords, original_shape, resize
        )
        for image_id in annotations["image_id"].unique().tolist()
    ]
    max_annotations = max(x[1].size(1) for x in coords_flags)
    coords_flags = [pad_coords(x, max_annotations) for x in coords_flags]
    coords = [x[0] for x in coords_flags]
    flags = [x[1] for x in coords_flags]
    return torch.stack(coords), torch.stack(flags)


def rearrange_classes(classes):
    distinct_classes = itertools.chain(*[list(x.values()) for x in classes])
    return {val: ix for ix, val in enumerate(distinct_classes, start=1)}


def collate_gt(tensor, original_classes, new_classes):
    for i in range(tensor.size(0)):
        for j in range(tensor.size(0)):
            tensor[i, j] = (
                0
                if tensor[i, j].item() == 0
                else new_classes[original_classes[tensor[i, j].item()]]
            )
    return tensor


def collate_mask(tensor, original_classes, new_classes):
    new_positions = [new_classes[x] - 1 for x in original_classes.values()]
    (
        m,
        c,
        h,
        w,
    ) = tensor.shape
    out = torch.zeros(size=(m, len(new_classes.keys()), h, w))
    out[:, new_positions, :, :] = tensor
    return out


def collate_bbox(bbox, flag, original_classes, new_classes, max_annotations):
    new_positions = [new_classes[x] - 1 for x in original_classes.values()]
    m, c, n, b_dim = bbox.shape
    out_bbox = torch.zeros(size=(m, len(new_classes.keys()), max_annotations, b_dim))
    out_flag = torch.full(
        size=(m, len(new_classes.keys()), max_annotations), fill_value=False
    )
    out_bbox[:, new_positions, :n, :] = bbox
    out_flag[:, new_positions, :n] = flag
    return out_bbox, out_flag


def collate_coords(coords, flag, original_classes, new_classes, max_annotations):
    new_positions = [new_classes[x] - 1 for x in original_classes.values()]
    m, c, n, k, c_dim = coords.shape
    out_coords = torch.zeros(
        size=(m, len(new_classes.keys()), max_annotations, k, c_dim)
    )
    out_flag = torch.full(
        size=(m, len(new_classes.keys()), max_annotations, k), fill_value=False
    )
    out_coords[:, new_positions, :n, :, :] = coords
    out_flag[:, new_positions, :n, :] = flag
    return out_coords, out_flag
