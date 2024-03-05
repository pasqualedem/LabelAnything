from datetime import datetime
import os
import pathlib
import cv2
import numpy as np
from pycocotools import mask as mask_utils
import xml.etree.ElementTree as ET
from scipy.ndimage import label, binary_dilation
from PIL import Image
import json
from tqdm import tqdm

url_voc = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
download_command = f"wget {url_voc}"
tar_command = f"tar -xf VOCtrainval_11-May-2012.tar"
instances_voc12 = {
    "info": {
        "description": "VOC 2012 Dataset Annotations files",
        "version": "1.0",
        "year": 2024,
        "contributor": "CILAB",
        "date_created": datetime.now().strftime("%Y-%m-%d"),
    },
    "images": [],
    "annotations": [],
    "categories": [],
}

VOC2012 = pathlib.Path("data/raw/VOCdevkit/VOC2012")


def get_items(root, ids):
    images = []
    all_boxes = []
    all_masks = []
    all_labels = []

    for image_id in tqdm(ids):
        image = _get_images(root, image_id)
        masks, boxes = _get_boxes_and_masks(root, image_id)
        labels = _get_label(root, image_id)

        images.append(image)
        all_boxes.append(boxes)
        all_masks.append(masks)
        all_labels.append(labels)

    return images, all_boxes, all_masks, all_labels


def _read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids


def _get_images(root, image_id):
    image_file = os.path.join(root, "JPEGImages", image_id + ".jpg")
    image = cv2.imread(str(image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _get_boxes_and_masks(root, image_id):
    mask_file = os.path.join(root, "SegmentationObject", image_id + ".png")
    mask_array = np.array(Image.open(mask_file))
    unique_values = np.unique(mask_array)
    masks = {}

    for value in unique_values:
        if value in [0, 255]:
            # If the value is 0 or 255, add it to the mask for 0
            _ = masks.get(0, np.zeros_like(mask_array)) | (mask_array == value)
        else:
            # Apply binary dilation before finding connected components
            dilated_mask = binary_dilation(mask_array == value)
            labeled_array, num_features = label(dilated_mask)
            for i in range(1, num_features + 1):
                masks[f"{value}_{i}"] = np.where(labeled_array == i, 1, 0)

    rle_masks = {}
    boxes = []

    for key, value in masks.items():
        rle = mask_utils.encode(np.asfortranarray(value.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")  # Convert bytes to string
        rle_masks[key] = rle

        bbox = mask_utils.toBbox(rle)
        boxes.append(bbox.tolist())

    return rle_masks, np.array(boxes, dtype=np.float32)


def _get_label(root, image_id):
    annotation_file = os.path.join(root, "Annotations", image_id + ".xml")
    objects = ET.parse(annotation_file).findall("object")
    labels = []

    for object in objects:
        class_name = object.find("name").text.lower().strip()
        labels.append(class_name)

    return np.array(labels)


def create_annotation(ids, images, boxes, rle_masks, labels, annotations):
    # generate set of categories
    annotations_images = []
    annotations_segmentations = []

    annotations_categories = [
        {"id": i, "name": name} for i, name in enumerate(set(np.concatenate(labels)))
    ]
    category_to_id = {
        category["name"]: category["id"] for category in annotations_categories
    }

    for enum, id_ in enumerate(ids):
        # print(ids[i])
        image = {
            "file_name": f"JPEGImages/{id_}.jpg",  # This is the only field that is compulsory
            "coco_url": f"JPEGImages/{id_}.jpg",
            "height": images[enum].shape[0],
            "width": images[enum].shape[1],
            "id": enum,
        }
        annotations_images.append(image)

    i = 0
    for enum, (box, rle, label) in enumerate(zip(boxes, rle_masks, labels)):
        for b, (_, rle_value), l in zip(box, rle.items(), label):
            annotation = {
                "segmentation": rle_value,
                "area": int(mask_utils.area(rle_value)),
                "image_id": enum,
                "bbox": b.tolist(),  # Assuming box is a list/array of [x_min, y_min, x_max, y_max]
                "category_id": category_to_id[l],
                "id": i,
            }
            annotations_segmentations.append(annotation)
            i += 1

    annotations["images"] = annotations_images
    annotations["annotations"] = annotations_segmentations
    annotations["categories"] = annotations_categories
    return annotations


def generate_dataset_file(input_folder):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    voc_folder = os.path.join(project_dir, input_folder)
    files = os.listdir(os.path.join(voc_folder, "ImageSets/Segmentation/"))
    contents = ""
    for file in files:
        with open(os.path.join(voc_folder, "ImageSets/Segmentation/", file), "r") as f:
            print(f"Reading {file}")
            file_content = f.read()
        contents += file_content

    with open(os.path.join(voc_folder, "ImageSets/Segmentation/dataset.txt"), "w") as f:
        f.write(contents)


def preprocess_voc(input_folder=None, output_folder=None):
    input_folder = input_folder or VOC2012
    output_folder = output_folder or pathlib.Path("data/annotations")

    if not os.path.exists(input_folder):
        print("Downloading VOC2012 dataset...")
        os.system(download_command)
        os.system(tar_command)
    else:
        print("VOC2012 dataset already exists!")

    if not os.path.exists(
        os.path.join(input_folder, "ImageSets/Segmentation/dataset.txt")
    ):
        print("Generating dataset file...")
        dataset = generate_dataset_file(input_folder)
    else:
        print("Dataset file already exists!")

    dataset = os.path.join(input_folder, "ImageSets/Segmentation/dataset.txt")

    ids = _read_image_ids(dataset)
    print(f"len ids: {len(ids)}")
    images, boxes, polygons, labels = get_items(input_folder, ids)
    annotations = create_annotation(
        ids,
        images,
        boxes,
        polygons,
        labels,
        instances_voc12,
    )

    with open(f"data/annotations/instances_voc12.json", "w") as f:
        json.dump(annotations, f)

    print("Done!")


if __name__ == "__main__":
    preprocess_voc()
