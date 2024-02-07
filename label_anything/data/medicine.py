import json
import os
import glob
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


data_dict = {"images": [], "annotations": [], "categories": []}


def download_and_extract_dataset(
    username="mateuszbuda",
    dataset_name="lgg-mri-segmentation",
    path="data/raw",
):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(f"{username}/{dataset_name}", unzip=True, path=path)
    print("Downloaded and extracted dataset:", dataset_name)


def generate_data(path):
    data_map = []
    if not path.endswith("/"):
        path += "/"
    for sub_dir_path in glob.glob(path + "*"):
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dirname, image_path])

    images = sorted([p for p in data_map[1::2] if "mask" not in p])
    masks = sorted([p for p in data_map[1::2] if "mask" in p])
    return images, masks


def generate_mask_bbox(mask_file) -> (dict, list):
    mask = np.array(Image.open(mask_file))
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # Convert bytes to string

    # Get the non-zero pixel coordinates
    y_indices, x_indices = np.nonzero(mask)

    # Check if there are any non-zero pixels
    if y_indices.size > 0 and x_indices.size > 0:
        # If there are non-zero pixels, calculate the bounding box
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    else:
        # If there are no non-zero pixels, return an empty bounding box
        bbox = None

    return rle, bbox


def generate_annotations(images, masks, annotations):
    annotations_images = []
    annotations_segmentations = []

    for idx, (image, mask) in enumerate(zip(images, masks)):
        image_name = image.split("/")[-1].split(".")[0]
        width, height = Image.open(image).size
        annotations_images.append(
            {
                "file_name": image_name,
                "url": image,
                "height": int(width),
                "width": int(height),
                "id": idx,
            }
        )

        rle, bbox = generate_mask_bbox(mask)
        if bbox is None:
            category_id = 0
            bbox = [0, 0, 0, 0]
        else:
            category_id = 1
            bbox = [int(b) for b in bbox]
        annotations_segmentations.append(
            {
                "segmentation": rle["counts"],
                "area": int(mask_utils.area(rle)),
                "image_id": idx,
                "bbox": bbox,
                "category_id": category_id,
                "id": idx,
            }
        )

    annotations["images"] = annotations_images
    annotations["annotations"] = annotations_segmentations
    annotations["categories"] = [
        {"id": 0, "name": "no_tumor"},
        {"id": 1, "name": "tumor"},
    ]

    return annotations


if __name__ == "__main__":
    if os.path.exists("data/raw/lgg-mri-segmentation"):
        print("Dataset already downloaded and extracted")
    else:
        download_and_extract_dataset()  # all parameters are set by default

    path = "data/raw/lgg-mri-segmentation/kaggle_3m/"
    images, masks = generate_data(path)
    data_dict = generate_annotations(images, masks, data_dict)
    with open("data/annotations/brain_mri.json", "w") as f:
        json.dump(data_dict, f)
    
    print("Done!")
