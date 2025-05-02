import re
import streamlit as st
from PIL import Image
import torch
from huggingface_hub import list_models

COLORS = [
        [0, 0, 0],         # Background
        [255, 0, 0],       # Class 2: Red
        [0, 0, 255],       # Class 4: Blue
        [255, 255, 0],     # Class 3: Yellow
        [0, 255, 0],       # Class 1: Green
        [255, 0, 255],     # Class 5: Magenta
        [0, 255, 255],     # Class 6: Cyan
        [192, 192, 192],   # Class 7: Light Gray
        [255, 0, 0],       # Class 8: Bright Red
        [0, 255, 0],       # Class 9: Bright Green
        [0, 0, 255],       # Class 10: Bright Blue
        [255, 255, 0],     # Class 11: Bright Yellow
        [255, 0, 255],     # Class 12: Bright Magenta
        [0, 255, 255],     # Class 13: Bright Cyan
        [128, 128, 128],   # Class 14: Dark Gray
        [255, 165, 0],     # Class 15: Orange
        [75, 0, 130],      # Class 16: Indigo
        [255, 20, 147],    # Class 17: Deep Pink
        [139, 69, 19],     # Class 18: Brown
        [154, 205, 50],    # Class 19: Yellow-Green
        [70, 130, 180],    # Class 20: Steel Blue
        [220, 20, 60],     # Class 21: Crimson
        [107, 142, 35],    # Class 22: Olive Drab
        [0, 100, 0],       # Class 23: Dark Green
        [205, 133, 63],    # Class 24: Peru
        [148, 0, 211],     # Class 25: Dark Violet
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

def debug_write(*args):
    st.write(*args)


def color_to_class(color):
    # color is in format rgba(r, g, b, a)
    # use regex to extract r, g, b (values can be float or int)
    color = re.findall(r"\d+\.\d+|\d+", color)
    color = [int(color[0]), int(color[1]), int(color[2])]
    return COLORS.index(color) - 1 # -1 because background is not a class


def get_color_from_class(classes, selected_class):
    selected_class_color = COLORS[classes.index(selected_class) + 1] # +1 because background is not a class
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
        key: value[idx].unsqueeze(0) if isinstance(value, torch.Tensor) else [value[idx]]
        for key, value in batch.items()
        }

class SupportExample(dict):
    img: Image
    prompts: dict
    reshape: tuple

    def __init__(
        self, support_image: Image, prompts: dict = {}, reshape: tuple = ()
    ):
        self.img = support_image
        self.prompts = prompts
        self.reshape = reshape
        self['img'] = support_image
        self['prompts'] = prompts
        self['reshape'] = reshape
        
    def draw(self, draw):
        raise NotImplementedError
    

DEFAULT_MODELS = [
    "pasqualedem/label_anything_sam_1024_coco",
    "pasqualedem/label_anything_256_sam_1024_coco", 
]
    
    
def retrieve_models():
    try:
        return [model.id for model in list_models(author="pasqualedem") if model.id.startswith("pasqualedem/label_anything")]
    except Exception as e:
        return DEFAULT_MODELS