import itertools
from einops import rearrange
import pandas as pd
import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_image_annotation import detection
from streamlit_drawable_canvas import st_canvas
from streamlit_tags import st_tags
from tempfile import NamedTemporaryFile


from label_anything.data.dataset import LabelAnythingDataset, VariableBatchSampler
from label_anything.data.transforms import CustomResize, CustomNormalize
from accelerate import Accelerator

import numpy as np
import torch
from torchvision.transforms import Compose, PILToTensor
from torch.utils.data import DataLoader
import numpy as np
import os

import lovely_tensors as lt
from label_anything.demo.preprocess import preprocess_to_batch
from label_anything.demo.utils import (
    COLORS,
    TEXT_COLORS,
    SupportExample,
    get_color_from_class,
    open_rgb_image,
    debug_write,
)
from label_anything.experiment.substitution import Substitutor
from label_anything.models.build_encoder import build_vit_b
from label_anything.utils.utils import ResultDict, load_yaml, torch_dict_load
from label_anything.models import model_registry

lt.monkey_patch()

from label_anything.models import build_lam_no_vit
from label_anything.data.examples import uniform_sampling
from label_anything.data import utils
from label_anything.data.utils import (
    AnnFileKeys,
    PromptType,
    BatchKeys,
    StrEnum,
    get_preprocess_shape,
)
from label_anything.experiment.utils import WrapperModule

from label_anything.demo.visualize import (
    load_from_wandb,
    plot_seg,
)
from label_anything.demo.builtin import built_in_dataset, predict

import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from easydict import EasyDict


LONG_SIDE_LENGTH = 1024
IMG_DIR = "data/coco/train2017"
ANNOTATIONS_DIR = "data/annotations/instances_val2017.json"
EMBEDDINGS_DIR = "data/coco/embeddings"
MAX_EXAMPLES = 30
VIT_B_SAM_PATH = "checkpoints/sam_vit_b_01ec64.pth"

SIZE = 1024

preprocess = Compose([CustomResize(SIZE), PILToTensor(), CustomNormalize(SIZE)])


class SS(StrEnum):
    SUPPORT_SET = "support_set"
    CLASSES = "classes"


@st.cache_resource
def get_data(_accelerator):
    dataset = LabelAnythingDataset(
        {
            "coco": {
                "name": "coco",
                "instances_path": st.session_state.get(
                    "annotations_dir", ANNOTATIONS_DIR
                ),
                "img_dir": st.session_state.get("img_dir", IMG_DIR),
                "preprocess": preprocess,
            }
        },
        {},
    )
    sampler = VariableBatchSampler(
        dataset,
        possible_batch_example_nums=[[1, 8], [1, 4], [1, 2], [1, 1]],
        num_processes=1,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn
    )
    dataloader = _accelerator.prepare(dataloader)
    return dataloader


@st.cache_resource
def load_model(_accelerator: Accelerator, run_id):
    folder = "best"
    model_file, config_file = load_from_wandb(run_id, folder)
    if config_file is not None:
        config = load_yaml(config_file)
        model_params = config["model"]
        name = model_params.pop("name")
    else:
        model_params = {}
        name = "lam_no_vit"
        st.warning(
            f"Config file not found, using default model params: {model_params}, {name}"
        )
    model = model_registry[name](**model_params)
    model = WrapperModule(model, None)
    model_state_dict = torch_dict_load(model_file)
    unmatched_keys = model.load_state_dict(model_state_dict, strict=False)
    model = _accelerator.prepare(model)
    if unmatched_keys.missing_keys:
        st.warning(f"Missing keys: {unmatched_keys.missing_keys}")
    if unmatched_keys.unexpected_keys:
        if unmatched_keys.unexpected_keys != [
            "loss.prompt_components.prompt_contrastive.t_prime",
            "loss.prompt_components.prompt_contrastive.bias",
        ]:
            st.warning(f"Unexpected keys: {unmatched_keys.unexpected_keys}")
    return model


@st.cache_resource
def load_image_encoder(image_encoder):
    checkpoint = VIT_B_SAM_PATH
    if not os.path.exists(checkpoint):
        st.warning(f"Checkpoint {checkpoint} not found")
        return None
    if image_encoder == "None":
        return None
    return build_vit_b(
        checkpoint=checkpoint,
        use_sam_checkpoint=True,
        project_last_hidden=False,
    ).cuda()


def reset_support(idx, image=False):
    if idx is None:
        st.session_state[SS.SUPPORT_SET] = {}
        return
    if idx in st.session_state[SS.SUPPORT_SET]:
        st.session_state[SS.SUPPORT_SET][idx].prompts = {}
        if image:
            st.session_state[SS.SUPPORT_SET].pop(idx)


def build_support_set():
    if st.session_state.get(SS.SUPPORT_SET, None) is None:
        st.session_state[SS.SUPPORT_SET] = {}
    st.write("Choose the classes you want to segment in the image")
    cols = st.columns(3)
    with cols[0]:
        classes = st_tags(
            label=SS.CLASSES,
            text="Type and press enter",
            value=st.session_state.get(SS.CLASSES, []),
            suggestions=["person", "car", "dog", "cat", "bus", "truck"],
        )
    with cols[1]:
        len_support_set = st.slider(
            "Number of support images", 1, 10, key="len_support_set"
        )
    with cols[2]:
        if st.button("Reset"):
            st.session_state[SS.SUPPORT_SET] = {}
    if len(classes) < len(st.session_state.get(SS.CLASSES, [])):  # Reset annotations
        for k in st.session_state[SS.SUPPORT_SET].keys():
            reset_support(k)
    if not classes:
        return
    st.session_state[SS.CLASSES] = classes
    st.write("## Upload and annotate the support images")
    i = 0
    tabs = st.tabs([f"Support Image {i+1}" for i in range(len_support_set)])
    for i in range(len_support_set):
        with tabs[i]:
            support_image = st.file_uploader(
                f"If you want, you can upload and annotate another support image",
                type=["png", "jpg"],
                on_change=lambda: reset_support(None, True),
                key=f"support_image_{i}",
            )
            if support_image is not None:
                add_support_image(support_image, i)
            i += 1


def add_support_image(support_image, idx):
    support_image = open_rgb_image(support_image)
    if idx not in st.session_state[SS.SUPPORT_SET]:
        st.write("Reset")
        st.session_state[SS.SUPPORT_SET][idx] = SupportExample(
            support_image=support_image
        )
    st.write(
        f"Use the annotation tool to annotate the image with bounding boxes, click Complete when you are done"
    )
    tab1, tab2 = st.tabs(["Annotate", "Load mask"])
    with tab1:
        cols = st.columns(3)
        with cols[0]:
            selected_class = st.selectbox(
                "Select the class you want to annotate",
                st.session_state[SS.CLASSES],
                key=f"selectbox_class_{idx}",
            )
        with cols[1]:
            prompt_type = st.selectbox(
                "Prompt Type",
                ["rect", "point", "polygon"],
                key=f"drawing_mode_{idx}",
            )
        with cols[2]:
            edit_mode = st.checkbox("Edit annotations", key=f"edit_mode_{idx}")
        edit_mode = prompt_type if not edit_mode else "transform"
        selected_class_color_f, selected_class_color_st = get_color_from_class(
            st.session_state[SS.CLASSES], selected_class
        )
        shape = get_preprocess_shape(
            support_image.size[1], support_image.size[0], LONG_SIDE_LENGTH
        )
        results = st_canvas(
            fill_color=selected_class_color_f,  # Fixed fill color with some opacity
            stroke_color=selected_class_color_st,  # Fixed stroke color with full opacity
            background_image=support_image,
            drawing_mode=edit_mode,
            key=f"input_prompt_detection_{idx}",
            width=shape[1],
            height=shape[0],
            stroke_width=2,
            update_streamlit=False,
        )
    with tab2:
        st.write("Load a mask to segment the image")
        st.write("Select the color for each class (background is always black)")
        color_map = {}
        color_cols = st.columns(len(st.session_state[SS.CLASSES]))
        for i, cls in enumerate(st.session_state[SS.CLASSES]):
            with color_cols[i]:
                color = st.selectbox(
                    f"Select color for {cls}",
                    TEXT_COLORS,
                    key=f"color_{idx}_{cls}",
                    index=i,
                )
                color_map[i] = np.array(COLORS[TEXT_COLORS.index(color)])
        mask = st.file_uploader(
            "Upload the mask", type=["png", "jpg"], accept_multiple_files=False
        )
        mask = np.array(open_rgb_image(mask)) if mask is not None else None
        st.image(mask, caption="Mask", use_column_width=True) if mask is not None else None
        if mask is not None:
            results = {
                "mask": mask,
                "color_map": color_map,
            }
    if results is not None:
        if hasattr(results, "json_data") and results.json_data is not None:
            st.write("Extracting prompts from canvas")
            st.session_state[SS.SUPPORT_SET][idx].prompts = results.json_data
            st.session_state[SS.SUPPORT_SET][idx].reshape = shape
        if isinstance(results, dict) and "mask" in results:
            st.write("Extracting prompts from mask")
            st.session_state[SS.SUPPORT_SET][idx].prompts = results
            st.session_state[SS.SUPPORT_SET][idx].reshape = shape


def try_it_yourself(model, image_encoder):
    st.write("Upload the image the you want to segment")
    query_images = st.file_uploader(
        "Choose an image", type=["png", "jpg"], accept_multiple_files=True
    )
    if len(query_images) > 0:
        images = [open_rgb_image(query_image) for query_image in query_images]
        with st.expander("Query Images"):
            cols = st.columns(len(query_images))
            for i, query_image in enumerate(query_images):
                image = open_rgb_image(query_image)
                with cols[i]:
                    # Save image in a temp file
                    st.image(image, caption=f"Query Image {i+1}", width=300)
    build_support_set()
    if (
        SS.SUPPORT_SET in st.session_state
        and SS.CLASSES in st.session_state
        and len(query_images) > 0
    ):
        batches = [
            preprocess_to_batch(
                image,
                st.session_state[SS.SUPPORT_SET],
                list(range(len(st.session_state[SS.CLASSES]))),
            )
            for image in images
        ]
        st.write(batches)
        debug_write(batches[0]["images"][0][0])
        if st.button("Predict"):
            progress = st.progress(0)
            tabs = st.tabs([f"Query Image {i+1}" for i in range(len(batches))])
            for i, batch in enumerate(batches):
                with tabs[i]:
                    result = predict(model, image_encoder, batch)
                    pred = result[ResultDict.LOGITS].argmax(dim=1)
                    st.json(result, expanded=False)
                    plots, titles = plot_seg(
                        batch,
                        pred,
                        COLORS,
                        dims=batch[BatchKeys.DIMS],
                        classes=st.session_state[SS.CLASSES],
                    )
                    cols = st.columns(2)
                    cols[0].image(plots[0], caption=titles[0], use_column_width=True)
                    cols[1].image(plots[1], caption=titles[1], use_column_width=True)
                    progress.progress((i + 1) / len(batches))


def main():
    st.set_page_config(layout="wide", page_title="Label Anything")
    st.title("Label Anything")
    st.sidebar.title("Settings")
    accelerator = Accelerator()
    with st.sidebar:
        # load model
        run_id = st.text_input("Run ID", "3ndl7had")
        model = load_model(accelerator, run_id).model  # WrapperModule
        image_encoder = st.selectbox("Image Encoder", options=["vit_sam_b"])
        image_encoder = load_image_encoder(image_encoder)
        st.divider()
        st.json(st.session_state, expanded=False)
    tiy_tab, dataset_tab = st.tabs(["Try it yourself", "Built-in dataset"])
    with tiy_tab:
        try_it_yourself(model, image_encoder)
    with dataset_tab:
        built_in_dataset(accelerator, model)


if __name__ == "__main__":
    main()
