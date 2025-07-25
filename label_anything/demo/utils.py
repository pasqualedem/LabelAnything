from copy import deepcopy
import os
import re
from einops import rearrange
import wget
import zipfile
from PIL import Image
import torch
from huggingface_hub import list_models
from nicegui import ui, run, observables
from label_anything.data import get_dataloaders

COLORS = [
    [0, 0, 0],  # Background
    [255, 0, 0],  # Class 2: Red
    [0, 0, 255],  # Class 4: Blue
    [255, 255, 0],  # Class 3: Yellow
    [0, 255, 0],  # Class 1: Green
    [255, 0, 255],  # Class 5: Magenta
    [0, 255, 255],  # Class 6: Cyan
    [192, 192, 192],  # Class 7: Light Gray
    [255, 0, 0],  # Class 8: Bright Red
    [0, 255, 0],  # Class 9: Bright Green
    [0, 0, 255],  # Class 10: Bright Blue
    [255, 255, 0],  # Class 11: Bright Yellow
    [255, 0, 255],  # Class 12: Bright Magenta
    [0, 255, 255],  # Class 13: Bright Cyan
    [128, 128, 128],  # Class 14: Dark Gray
    [255, 165, 0],  # Class 15: Orange
    [75, 0, 130],  # Class 16: Indigo
    [255, 20, 147],  # Class 17: Deep Pink
    [139, 69, 19],  # Class 18: Brown
    [154, 205, 50],  # Class 19: Yellow-Green
    [70, 130, 180],  # Class 20: Steel Blue
    [220, 20, 60],  # Class 21: Crimson
    [107, 142, 35],  # Class 22: Olive Drab
    [0, 100, 0],  # Class 23: Dark Green
    [205, 133, 63],  # Class 24: Peru
    [148, 0, 211],  # Class 25: Dark Violet
]
TEXT_COLORS = [
    "Black",
    "Red",
    "Blue",
    "Yellow",
    "Green",
    "Magenta",
    "Cyan",
    "Light Gray",
    "Bright Red",
    "Bright Green",
    "Bright Blue",
    "Bright Yellow",
    "Bright Magenta",
    "Bright Cyan",
    "Dark Gray",
    "Orange",
    "Indigo",
    "Deep Pink",
    "Brown",
    "Yellow-Green",
    "Steel Blue",
    "Crimson",
    "Olive Drab",
    "Dark Green",
    "Peru",
    "Dark Violet",
]


def debug_write(*args, **kwargs):
    """A simple debug function to write messages to the console."""
    print(*args, **kwargs)


def color_to_class(color):
    # color is in format rgba(r, g, b, a)
    # use regex to extract r, g, b (values can be float or int)
    color = re.findall(r"\d+\.\d+|\d+", color)
    color = [int(color[0]), int(color[1]), int(color[2])]
    return COLORS.index(color) - 1  # -1 because background is not a class


def get_color_from_class(classes, selected_class):
    selected_class_color = COLORS[
        classes.index(selected_class) + 1
    ]  # +1 because background is not a class
    selected_class_color_f = f"rgba({selected_class_color[0]}, {selected_class_color[1]}, {selected_class_color[2]}, 0.3)"
    selected_class_color_st = f"rgba({selected_class_color[0]}, {selected_class_color[1]}, {selected_class_color[2]}, 0.8)"
    return selected_class_color_f, selected_class_color_st


canvas_to_prompt_type = {
    "circle": "point",
    "rect": "bbox",
    "path": "mask",
}


def open_rgb_image(path):
    img = Image.open(path)
    img = img.convert("RGB")
    return img


def take_elem_from_batch(batch, idx):
    return {
        key: (
            value[idx].unsqueeze(0) if isinstance(value, torch.Tensor) else [value[idx]]
        )
        for key, value in batch.items()
    }


class SupportExample(dict):
    img: Image
    prompts: dict
    reshape: tuple

    def __init__(self, support_image: Image, prompts: dict = {}, reshape: tuple = ()):
        self.img = support_image
        self.prompts = prompts
        self.reshape = reshape
        self["img"] = support_image
        self["prompts"] = prompts
        self["reshape"] = reshape

    def draw(self, draw):
        raise NotImplementedError


DEFAULT_MODELS = [
    "pasqualedem/label_anything_sam_1024_coco",
    "pasqualedem/label_anything_256_sam_1024_coco",
]


def retrieve_models():
    try:
        return [
            model.id
            for model in list_models(author="pasqualedem")
            if model.id.startswith("pasqualedem/label_anything")
        ]
    except Exception as e:
        return DEFAULT_MODELS


COCO_PARAMS = {
    "name": "coco",
    "split": "val",
    "val_fold_idx": 3,
    "n_folds": 4,
    "n_shots": 1,
    "n_ways": 1,
    "do_subsample": False,
    "add_box_noise": False,
    "val_num_samples": 100,
}
COCO_NAME = "val_coco20i"
COCO_IMG_DIR = "data/coco/train_val_2017"
COCO_INSTANCES_PATH = "data/coco/annotations/instances_val2014.json"



parameters = {
    "dataloader": {
        "num_workers": 0,
        "possible_batch_example_nums": [[1]],
        "val_possible_batch_example_nums": [[1]],
        "prompt_types": ["mask"],
        "prompt_choice_level": ["episode"],
        "val_prompt_types": ["mask"],
    },
    "dataset": {
        "preprocess": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "image_size": 480,
        },
        "datasets": {
            COCO_NAME: COCO_PARAMS,
        },
        "common": {"remove_small_annotations": True},
    },
}

async def download_coco_instances(path):
   url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

   wget.download(url, path)
   # Unzip the downloaded file
   with zipfile.ZipFile(path, 'r') as zip_ref:
      zip_ref.extractall(os.path.dirname(path))
   os.remove(path)
   ui.notify("COCO instances downloaded and extracted", color="green")


async def get_data(
    n_ways,
    n_shots,
    n_examples,
    image_size,
    custom_preprocess,
    all_example_categories,
    prompt_types,
    max_points,
    fold,
    class_based_sampling=False,
):
    # if os.path.exists(COCO_IMG_DIR):
    #     parameters["dataset"]["datasets"][COCO_NAME]["img_dir"] = COCO_IMG_DIR
    
    if not os.path.exists(COCO_INSTANCES_PATH):
       ui.notify("COCO instances not found, downloading from official COCO repository...", color="red")
       os.makedirs(os.path.dirname(COCO_INSTANCES_PATH), exist_ok=True)
       run.io_bound(download_coco_instances, COCO_INSTANCES_PATH)

    parameters["dataset"]["datasets"][COCO_NAME]["instances_path"] = COCO_INSTANCES_PATH
    parameters["dataset"]["datasets"][COCO_NAME]["n_ways"] = n_ways
    parameters["dataset"]["datasets"][COCO_NAME]["n_shots"] = n_shots
    parameters["dataset"]["datasets"][COCO_NAME]["n_examples"] = n_examples
    parameters["dataset"]["datasets"][COCO_NAME]["image_size"] = image_size
    parameters["dataset"]["datasets"][COCO_NAME]["val_fold_idx"] = fold
    parameters["dataset"]["datasets"][COCO_NAME][
        "class_based_sampling"
    ] = class_based_sampling
    parameters["dataset"]["preprocess"]["image_size"] = image_size
    parameters["dataset"]["common"]["custom_preprocess"] = custom_preprocess
    parameters["dataset"]["common"]["image_size"] = image_size
    parameters["dataset"]["common"]["max_points_per_annotation"] = max_points
    parameters["dataset"]["common"]["all_example_categories"] = all_example_categories
    print("Parameters for COCO20i dataset defined")
    if not prompt_types:
        ui.notify("Please select at least one prompt type.", color="red")
        return
    parameters["dataloader"]["prompt_types"] = prompt_types
    parameters["dataloader"]["val_prompt_types"] = prompt_types
    _, val, _ = get_dataloaders(
        deepcopy(parameters["dataset"]),
        deepcopy(parameters["dataloader"]),
        num_processes=1,
    )
    return val[COCO_NAME]


def run_computation(model, batch, device):
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model = model.to(device)
    
    with torch.no_grad():
        result = model(batch)
    return {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in result.items()
    }


def get_features(model, batch, device):
    b, n = batch.shape[:2]
    batch = rearrange(batch, "b n c h w -> (b n) c h w")
    model = model.to(device)
    with torch.no_grad():
        result = torch.cat(
            [model(batch[i].unsqueeze(0).to(device)) for i in range(batch.shape[0])], dim=0
        )
    result = rearrange(result, "(b n) c h w -> b n c h w", b=b)
    return result.cpu()


def sanitize(l):
    if isinstance(l, observables.ObservableList):
        return [sanitize(elem) for elem in l]
    elif isinstance(l, list):
        return [sanitize(elem) for elem in l]
    elif isinstance(l, observables.ObservableDict):
        return {k: sanitize(v) for k, v in l.items()}
    elif isinstance(l, dict):
        return {k: sanitize(v) for k, v in l.items()}
    elif isinstance(l, observables.ObservableSet):
        return {sanitize(elem) for elem in l}
    elif isinstance(l, set):
        return {sanitize(elem) for elem in l}
    else:
        return l