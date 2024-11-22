import re
import streamlit as st
from PIL import Image
import torch
from huggingface_hub import list_models

COLORS = [
    # blue
    (0, 0, 255),
    # red
    (255, 0, 0),
    # green
    (0, 255, 0),
    # yellow
    (255, 255, 0),
    # purple
    (255, 0, 255),
    # cyan
    (0, 255, 255),
    # orange
    (255, 165, 0),
    # pink
    (255, 192, 203),
    # brown
    (139, 69, 19),
    # grey
    (128, 128, 128),
    # black
    (0, 0, 0),
    # white
    (255, 255, 255),
]
TEXT_COLORS = [
    "blue",
    "red",
    "green",
    "yellow",
    "purple",
    "cyan",
    "orange",
    "pink",
    "brown",
    "grey",
    "black",
    "white",
]

def debug_write(*args):
    st.write(*args)


def color_to_class(color):
    # color is in format rgba(r, g, b, a)
    # use regex to extract r, g, b (values can be float or int)
    color = re.findall(r"\d+\.\d+|\d+", color)
    color = (int(color[0]), int(color[1]), int(color[2]))
    return COLORS.index(color) - 1 # -1 because background is not a class


def get_color_from_class(classes, selected_class):
    selected_class_color = COLORS[classes.index(selected_class) + 1] # +1 because background is not a class
    selected_class_color_f = f"rgba({selected_class_color[0]}, {selected_class_color[1]}, {selected_class_color[2]}, 0.3)"
    selected_class_color_st = f"rgba({selected_class_color[0]}, {selected_class_color[1]}, {selected_class_color[2]}, 0.8)"
    selected_class_color_focused = f"rgba({selected_class_color[0]}, {selected_class_color[1]}, {selected_class_color[2]}, 1)"
    return selected_class_color_f, selected_class_color_st, selected_class_color_focused


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