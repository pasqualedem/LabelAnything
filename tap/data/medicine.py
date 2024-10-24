import json
import os
import glob
import random
import shutil
import numpy as np
from datetime import datetime
from PIL import Image
from pycocotools import mask as mask_utils
from kaggle.api.kaggle_api_extended import KaggleApi


data_dict = {
    "info": {
        "description": "Brain MRI Dataset Annotations files",
        "version": "1.0",
        "year": 2024,
        "contributor": "CILAB",
        "date_created": datetime.now().strftime("%Y-%m-%d"),
    },
    "images": [],
    "annotations": [],
    "categories": [],
}


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
        image = image.replace("data/raw/lgg-mri-segmentation/kaggle_3m/", "")
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
                "segmentation": rle,
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
        {"id": 0, "name": "background"},
        {"id": 1, "name": "tumor"},
    ]

    return annotations


def read_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith("_mask.tif") and file.endswith(".tif"):
                file_list.append(os.path.join(root, file))
    return file_list


def create_dict_image_mask(data_dir):
    files = read_files_in_folder(data_dir)
    image_mask = {}
    for f in files:
        mask_file = f[:-4] + "_mask.tif"
        image_mask[f] = mask_file
    return image_mask


def split_train_test(data_dir, train_dir, test_dir, test_ratio):
    image_mask_dict = create_dict_image_mask(data_dir)
    items = list(image_mask_dict.items())
    np.random.shuffle(items)
    num_test = int(len(items) * test_ratio)
    test_items = items[:num_test]
    train_items = items[num_test:]
    
    for image_file, mask_file in test_items:
        shutil.move(image_file, os.path.join(test_dir, os.path.basename(image_file)))
        shutil.move(mask_file, os.path.join(test_dir, os.path.basename(mask_file)))

    for image_file, mask_file in train_items:
        shutil.move(image_file, os.path.join(train_dir, os.path.basename(image_file)))
        shutil.move(mask_file, os.path.join(train_dir, os.path.basename(mask_file)))


if __name__ == "__main__":
    if os.path.exists("data/raw/lgg-mri-segmentation"):
        print("Dataset already downloaded and extracted")
    else:
        download_and_extract_dataset()  # all parameters are set by default

    path = "data/raw/lgg-mri-segmentation/kaggle_3m/"
    images, masks = generate_data(path)
    # data_dict = generate_annotations(images, masks, data_dict)
    # with open("data/annotations/brain_mri.json", "w") as f:
    #     json.dump(data_dict, f)

    # Usa la funzione per suddividere la tua struttura di cartelle    
    data_dir = "/home/emanuele/LabelAnything/data/raw/lgg-mri-segmentation/kaggle_3m"
    train_dir = "/home/emanuele/LabelAnything/data/brain/train"
    test_dir = "/home/emanuele/LabelAnything/data/brain/test"
    split_train_test(data_dir, train_dir, test_dir, 0.2)
    print("Done!")
