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


def split_train_test(input_folder, train_ratio=0.8):
    # Controlla se il percorso di output esiste, altrimenti lo crea
    train_folder = os.path.join(input_folder, "train")
    test_folder = os.path.join(input_folder, "test")
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Itera su tutte le sottocartelle nell'input_folder
    for root, dirs, files in os.walk(input_folder):
        # Escludi la cartella di output
        if root == train_folder or root == test_folder:
            continue

        # Crea la corrispondente struttura di cartelle nell'output_folder
        relative_path = os.path.relpath(root, input_folder)
        train_path = os.path.join(train_folder, relative_path)
        test_path = os.path.join(test_folder, relative_path)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        # Ottieni la lista dei file nella cartella corrente
        files = [f for f in files if not f.startswith(".")]  # Escludi i file nascosti

        # Scegli casualmente i file per il set di allenamento e test
        num_train = int(len(files) * train_ratio)
        train_files = random.sample(files, num_train)
        test_files = [f for f in files if f not in train_files]

        # Copia i file nella rispettiva cartella di train e test
        for file in train_files:
            shutil.copy(os.path.join(root, file), os.path.join(train_path, file))
        for file in test_files:
            shutil.copy(os.path.join(root, file), os.path.join(test_path, file))


def read_files_in_folder(folder_path):
    file_list = []

    # Itera su tutte le sottocartelle e i file nella cartella
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Crea il percorso completo del file e aggiungilo alla lista
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    print(len(file_list))
    


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
    # split_train_test(path, train_ratio=0.8)
    read_files_in_folder(os.path.join(path, 'train'))
    print("Done!")
