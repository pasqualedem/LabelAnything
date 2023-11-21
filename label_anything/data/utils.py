import glob
import pandas as pd
import torchvision.transforms
from PIL import Image, ImageDraw
from itertools import combinations

import torch
import pickle
import numpy as np
import itertools
import json
from typing import List, Tuple, Dict


MAX_PIXELS_BBOX_NOISE = 20  # noise limit for bounding boxes taken from SA


def get_max_annotations(annotations):
    anns = []
    for image in annotations:
        for cat in annotations[image]:
            anns += [annotations[image][cat].shape[0]]
    return max(anns)

def compute_j_index(class_a: List[int], class_b: List[int]) -> float:
    """
    Computes the Jaccard Index given two different class sets.

    Arguments:
        class_a (List[int]): list of integers, representing the classes of element a.
        class_b (List[int]): list of integers, representing the classes of element b.

    Returns:
        float: Jaccard index.
    """
    class_a = set(class_a)
    class_b = set(class_b)
    return len(class_a.intersection(class_b)) / len(class_a.union(class_b))


def compute_j_index_n_sets(sets):
    return len(set.intersection(*sets)) / len(set.union(*sets))


def mean_pairwise_j_index(sets):
    return sum(compute_j_index(a, b) for a, b in combinations(sets, 2)) / (len(sets) * (len(sets) - 1))


def convert_polygons(polygons: List[List[float]]) -> List[List[Tuple[int]]]:
    """
    Converts a list of polygons, feasible for considering it as a list of coordinates, representing the vertices of a polygon.

    Arguments:
        polygons (List[List[float]]): list of polygons to be formatted

    Returns:
        List[List[Tuple[int]]]: list of polygons represented by a couple of vertices.
    """
    return [[(int(pol[i]), int(pol[i + 1])) for i in range(0, len(pol), 2)] for pol in polygons]


def get_mask(
        img_shape: torch.Size,
        reshape: torchvision.transforms.Resize,
        segmentations: List[List[Tuple[int]]]
) -> torch.Tensor:
    """
    Transforms a polygon to a matrix mask.

    Arguments:
        img_shape: original shape of the image
        reshape: reshape transformation
        segmentations: segmentations expressed by polygons

    Returns:
        torch.Tensor: single mask of shape HxW. Each pixel is 1 if it is part of the annotations, and 0 if not.
    """
    if segmentations == [[]]:  # for empty segmentation
        return reshape(torch.zeros(img_shape))[0], torch.Tensor([0.0])
    image = Image.new('L', img_shape[1:][::-1], 0)  # due to problem with shapes
    draw = ImageDraw.Draw(image)
    for pol in convert_polygons(segmentations):
        draw.polygon(pol, outline=1, fill=1)
    mask = np.asarray(image)
    mask = torch.Tensor(mask)
    return reshape(mask.unsqueeze(dim=0)), torch.Tensor([1.0])


def get_mask_per_image(
        annotations: pd.DataFrame,
        image_id: int,
        image_shape: torch.Size,
        reshape: torchvision.transforms.Resize,
        target_classes: List[int]
) -> torch.Tensor:
    """
    Transforms all the segmentations associated to the given image into masks, given a list of classes.

    Arguments:
        annotations: DataFrame used to handle the annotations.
        image_id: the image to focus on.
        image_shape: original image shape.
        reshape: reshape transformation.
        target_classes: list of classes to focus on.

    Returns:
        torch.Tensor: masks batched on classes, with shape C x H x W
    """
    masks_flags = [get_mask(image_shape, reshape, itertools.chain(*annotations[(annotations.image_id == image_id) & (annotations.category_id == x)].segmentation.tolist()))
                    for x in target_classes]
    return torch.stack([x[0] for x in masks_flags]), torch.stack([x[1] for x in masks_flags])


def get_prompt_mask(
        annotations: pd.DataFrame,
        image_shape: List[torch.Size],
        reshape: torchvision.transforms.Resize,
        target_classes: List[int]
) -> torch.Tensor:
    """
    Computes all the prompt masks for a given query images.

    Arguments:
        annotations: DataFrame containing all the annotations useful for the mask generation.
        image_shape: list of all the original shape for each example image.
        reshape: reshape transformation.
        target_classes: classes to consider.

    Returns:
        torch.Tensor: masks batched for all the example images of shape MxCxHxW
    """
    masks_flags = [get_mask_per_image(annotations, x, image_shape[ix], reshape, target_classes)
        for ix, x in enumerate(annotations.image_id.unique().tolist())]
    return torch.stack([x[0] for x in masks_flags]).squeeze(dim=2), torch.stack([x[1] for x in masks_flags]).squeeze(dim=2)


def add_noise(
        bbox: List[float]
) -> List[float]:
    """
    Adds random noise to a bounding box. Each coordinate is perturbed by adding x ~ N(0, std), where std is
    0.1 * abs(x2-x1) or 0.1 * abs(y2 - y1). however, if this number is grater than 20, it is truncated at 20 pixels.
    """
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


def get_bboxes(
        bboxes_entries: pd.DataFrame,
        image_id: int,
        category_id: int,
        len_bbox: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the bounding boxes, given an image and a class.

    Arguments:
        bboxes_entries: DataFrame of annotations for the bounding boxes.
        image_id: id of te given image.
        category_id: id of the given class.
        len_bbox: maximum number of bounding boxes (useful for padding).

    Returns:
        torch.Tensor: bounding box tensor of shape N x 4
        torch.Tensor: flag tensor of shape N, indicating whether the ith bounding box is real or a pad one.
    """
    bbox = bboxes_entries[(bboxes_entries.image_id == image_id) & (bboxes_entries.category_id == category_id)][
        'bbox'].tolist()  # extracting the bounding boxes
    bbox = list(map(lambda x: add_noise(x), bbox))  # [x, y, w, h] -> [x1, y1, x2, y2]
    bbox = torch.Tensor(bbox)

    # padding
    ans = torch.cat([
        bbox, torch.zeros((len_bbox - bbox.size(0), 4))
    ], dim=0)
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


def get_prompt_bbox_per_image(
        bbox_entries: pd.DataFrame,
        img_id: int,
        target_classes: List[int],
        max_anns: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the bounding boxes for a single image.

    Arguments:
        bbox_entries: DataFrame of annotations.
        img_id: id of given image.
        target_classes: list of target classes on which filter the annotations.
        max_anns: maximum number of annotations (useful for padding).

    Returns:
        torch.Tensor: tensor of bounding boxes, batched for the class. Tensor shape: C x N x 4.
        torch.Tensor: tensor of bounding boxes flag, batched for the class. tensor shape: C x N.
    """
    res = [get_bboxes(bbox_entries, img_id, x, max_anns) for x in target_classes]  # do it for each class
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def get_prompt_bbox(
        bbox_entries: pd.DataFrame,
        target_classes: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates all the bounding boxes for each example image.

    Arguments:
        bbox_entries: dataframe of annotations.
        target_classes: list of classes, representing the classes of the query image.

    Returns:
        torch.Tensor: tensor of bounding boxes, batched for example images. Tensor shape: M x C x N x 4.
        torch.Tensor: tensor of bounding boxes flag, batched for example images. tensor shape: M x C x N.
    """
    max_anns = get_max_bbox(bbox_entries)
    res = [
        get_prompt_bbox_per_image(bbox_entries, x, target_classes, max_anns)
        for x in bbox_entries.image_id.unique().tolist()
    ]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def get_max_bbox(
        annotations: pd.DataFrame
) -> int:
    """
    Calculates the maximum number of annotations present in ``annotations``, grouped by images and category id.
    """
    return max(
        len(
            annotations[
                (annotations.image_id == img) & (annotations.category_id == cat)
            ]
        )
        for img in annotations.image_id.unique()
        for cat in annotations.category_id.unique()
    )


def get_gt(
        annotations: pd.DataFrame,
        image_shape: torch.Size,
        target_classes: List[int]
) -> torch.Tensor:
    """
    Computes the target segmentations, represented as a matrix of integers.

    Arguments:
        annotations: dataframe of annotations.
        image_shape: original shape of the query image.
        target_classes: list of target classes to focus on.

    Returns:
        torch.Tensor: segmentation mask of shape H x W. each value of this matrix represents the membership class.
    """
    gt = Image.new('L', image_shape[1:][::-1], 0)
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


def load_dict(path: str):
    """
    Loads a dictionary from file.
    """
    name, ext = str(path).split(".")
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


def load_instances(
        path: str
) -> Dict:
    """
    Loads the instances from file.
    """
    if "*" in str(path):
        files = glob.glob(path)
        instances = {}
        for file in files:
            instances.update(load_dict(file))
    else:
        instances = load_dict(path)
    return instances


def get_coords(
        annotation: pd.Series,
        num_coords: int,
        original_shape: torch.Size,
        reshape: torchvision.transforms.Resize
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates ``num_coords`` points , giving a single annotation.

    Arguments:
        annotation: raw annotation.
        num_coords: number of coordinates to extract.
        original_shape: image original shape
        reshape: reshape transformation.

    Returns:
        torch.Tensor: tensor of coordinates of shape K x 2.
        torch.Tensor: flag tensor of shape K.
    """
    if annotation["segmentation"] == [[]]:  # empty segmentation
        return torch.Tensor([0.0, 0.0]), torch.Tensor([False])

    # get the mask with respect to the annotation
    mask = get_mask(img_shape=original_shape, reshape=reshape, segmentations=annotation["segmentation"])[0].squeeze(dim=0)
    coords = torch.nonzero(mask == 1)  # get all the possible coordinates

    # pad the candidates if they are too few
    if coords.size(0) < num_coords:  # for very small masks
        return torch.cat([coords, torch.zeros(size=((num_coords - coords.size(0)), *coords.shape[1:]))]), \
               torch.cat([torch.full(size=(coords.size(0),), fill_value=True),
                          torch.full(size=((num_coords - coords.size(0)),), fill_value=False)])

    # extract the candidates
    indexes = np.random.randint(0, coords.size(0), size=num_coords).tolist()
    return coords[indexes], torch.full(size=(num_coords,), fill_value=True)


def get_coords_per_image_class(
        annotations: pd.DataFrame,
        image_id: int,
        class_id: int,
        num_coords: int,
        original_shape: torch.Size,
        reshape: torchvision.transforms.Resize
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts all the coordinates needed for all the annotations, given an image and a class.

    Arguments:
        annotations: DataFrame of raw annotations.
        image_id: id of the given image.
        class_id: id of the given class.
        num_coords: number of coordinates to extract for a single given image-class annotation.
        original_shape: original image shape
        reshape: reshape transformation.

    Returns:
        torch.Tensor: tensor of coordinates of shape N x K x 2.
        torch.Tensor: tensor of flag of shape N x K.
    """
    target_annotations = annotations[(annotations.image_id == image_id) & (annotations.category_id == class_id)]
    if len(target_annotations) == 0:
        return torch.zeros(size=(1, num_coords, 2)), torch.full(size=(1, num_coords),
                                                                fill_value=False)
    coords_flags = [get_coords(row, num_coords, original_shape, reshape) for _, row in
                    target_annotations.iterrows()]
    coords = [x[0] for x in coords_flags]
    flags = [x[1] for x in coords_flags]
    return torch.stack(coords), torch.stack(flags)


def pad_coords_image(
        coords_flags: Tuple[torch.Tensor, torch.Tensor],
        max_annotations: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads coordinates tensor and its associated flag tensor, according to the maximum number of annotations.
    """
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
        annotations: pd.DataFrame,
        image_id: int,
        target_classes: List[int],
        num_coords: int,
        original_shape: torch.Size,
        resize: torchvision.transforms.Resize
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the coordinates of a given image, considering a pool of classes.
    For each class, the method retrieves all the annotations for the image corresponding to the i-th class
    and extract a certain number of points, if possible. To handle the case in which the image has no annotation
    of a given class, there is a pad tensor, in which a single element indicates whether the point is real or a pad one.

    Arguments:
        annotations: raw annotations.
        image_id: the given image id.
        target_classes: list of class ids.
        num_coords: the number of coordinates to extract for each annotation.
        original_shape: original image shape.
        resize: resize transformation.

    Returns:
        torch.Tensor: coord tensor of shape C x N x K x 2.
        torch.Tensor: flag tensor of shape C x N x K.
    """
    coords_flags = [get_coords_per_image_class(annotations, image_id, class_id, num_coords, original_shape, resize)
                    for class_id in target_classes]
    max_annotations = max([x[0].shape[0] for x in coords_flags])
    padded = [pad_coords_image(x, max_annotations) for x in coords_flags]
    coords = [x[0] for x in padded]
    flags = [x[1] for x in padded]
    return torch.stack(coords), torch.stack(flags)


def pad_coords(
        coords_flags: Tuple[torch.Tensor, torch.tensor],
        max_annotations: int
) -> Tuple[torch.Tensor, torch.tensor]:
    """
    Pads coordinates tensor and its associated flag tensor, according to the maximum number of annotations.
    """
    coords, flags = coords_flags
    if coords.size(1) == max_annotations:
        return coords, flags
    coords = coords.permute(1, 0, 2, 3)
    flags = flags.permute(1, 0, 2)
    return torch.cat([coords, torch.zeros(size=((max_annotations - coords.size(0)), *coords.shape[1:]))]).permute(1, 0, 2, 3), \
           torch.cat([flags, torch.full(size=((max_annotations - flags.size(0)), *flags.shape[1:]), fill_value=False)]).permute(1, 0, 2)


def get_prompt_coords(
        annotations: pd.DataFrame,
        target_classes: List[int],
        num_coords: int,
        original_shape: List[torch.Size],
        resize: torchvision.transforms.Resize
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the prompt coordinates for a pool of example images and its associated flag tensor.

    Arguments:
        annotations: raw annotations.
        target_classes: list of class ids.
        num_coords: number of coordinates to generate for each annotation.
        original_shape: list of size for each example image
        resize: resize transformation.

    Returns:
        torch.Tensor: coords tensor of shape M x C x N x K x 2.
        torch.Tensor: flag tensor of shape M x C x N x K.
    """
    coords_flags = [get_coords_per_image(annotations, image_id, target_classes, num_coords, original_shape[ix], resize)
                    for ix, image_id in enumerate(annotations["image_id"].unique().tolist())]
    max_annotations = max(x[1].size(1) for x in coords_flags)
    coords_flags = [pad_coords(x, max_annotations) for x in coords_flags]
    coords = [x[0] for x in coords_flags]
    flags = [x[1] for x in coords_flags]
    return torch.stack(coords), torch.stack(flags)


# ------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------COLLATE FUNCTIONS----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #


def rearrange_classes(
        classes: List[Dict[int, int]]
) -> Dict[int, int]:
    """
    Returns a new dict for class positions in a batch
    """
    distinct_classes = set(itertools.chain(*[list(x.values()) for x in classes]))
    return {val: ix for ix, val in enumerate(distinct_classes, start=1)}


def collate_gt(
        tensor: torch.Tensor,
        original_classes: Dict[int, int],
        new_classes: Dict[int, int]
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
            tensor[i, j] = 0 if tensor[i, j].item() == 0 else new_classes[original_classes[tensor[i, j].item()]]
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
    m, c, h, w, = masks.shape
    out = torch.zeros(size=(m, num_classes, h, w))
    out_flags = torch.zeros(size=(m, num_classes))
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
    """Collate ground truths for a single sample (query + support).
    """
    out = torch.zeros(dims)
    dim0, dim1 = gt.size()
    out[:dim0, :dim1] = gt
    return out.type(torch.uint8)


def collate_batch_gts(gt, dims, fill_value=-100):
    """Collate ground truths for a batch of samples, here the fill_value must be -100.
    """
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