import itertools
from einops import rearrange
import pandas as pd
import streamlit as st
import torch
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from label_anything import LabelAnything
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

from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization
)

from label_anything.demo.preprocess import denormalize, preprocess_support_set, preprocess_to_batch
from label_anything.demo.utils import (
    COLORS,
    TEXT_COLORS,
    SupportExample,
    get_color_from_class,
    open_rgb_image,
    debug_write,
    retrieve_models,
    take_elem_from_batch,
)
from label_anything.experiment.substitution import Substitutor
from label_anything.models.build_encoder import build_vit_b, build_vit_b_mae
from label_anything.models.explainer import LamExplainer
from label_anything.utils.utils import ResultDict, load_yaml, torch_dict_load
from label_anything.models import model_registry

lt.monkey_patch()

from label_anything.models import build_lam_no_vit
from label_anything.data.examples import uniform_sampling
from label_anything.data import get_preprocessing, utils
from label_anything.data.utils import (
    AnnFileKeys,
    PromptType,
    BatchKeys,
    StrEnum,
    get_preprocess_shape,
)
from label_anything.experiment.utils import WrapperModule

from label_anything.demo.visualize import (
    draw_all,
    draw_masks,
    get_image,
    load_from_wandb,
    plot_seg,
)
from label_anything.demo.builtin import built_in_dataset, predict

import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
import torchvision.transforms as TvT
from easydict import EasyDict


IMG_DIR = "data/coco/train2017"
ANNOTATIONS_DIR = "data/annotations/instances_val2017.json"
EMBEDDINGS_DIR = "data/coco/embeddings"
MAX_EXAMPLES = 30
VIT_B_SAM_PATH = "checkpoints/sam_vit_b_01ec64.pth"

SIZE = 384
PROMPT_SIZE = 512
CUSTOM_PREPROCESS = False

dataset_params = {
    "common": {
        "remove_small_annotations": True,
        "image_size": SIZE,
        "custom_preprocess": False
    }
}

preprocess = get_preprocessing(dataset_params)


def load_dcama():
    return model_registry['dcama'](
    backbone_checkpoint="checkpoints/swin_base_patch4_window12_384.pth",
    model_checkpoint="checkpoints/swin_fold3.pt"
)


class SS(StrEnum):
    SUPPORT_SET = "support_set"
    CLASSES = "classes"
    RESULT = "result"


@st.cache_resource
def load_model(device):
    return load_dcama().to(device)


def reset_support(idx):
    if idx is None:
        st.session_state[SS.SUPPORT_SET] = []
        st.session_state[SS.CLASSES] = []
        return
    st.session_state[SS.SUPPORT_SET].pop(idx)


def build_support_set():
    if st.session_state.get(SS.SUPPORT_SET, None) is None:
        st.session_state[SS.SUPPORT_SET] = []
    st.write("Choose the classes you want to segment in the image")
    cols = st.columns(2)
    with cols[0]:
        new_class = st.text_input("Type and press enter to add a class")
        classes = st.session_state.get(SS.CLASSES, [])
        if new_class not in classes and new_class != "":
            classes.append(new_class)
        st.session_state[SS.CLASSES] = classes
    with cols[1]:
        if st.button("Reset"):
            reset_support(None)
            classes = []
    if not classes:
        return
    st.write("Classes:", ", ".join(classes))
    st.write("## Upload and annotate the support images")

    support_image = st.file_uploader(
        "If you want, you can upload and annotate another support image",
        type=["png", "jpg", "webp"],
        key="support_image",
    )
    if support_image is not None:
        add_support_image(support_image)


def add_support_image(support_image):
    support_image = open_rgb_image(support_image)
    st.write(
        "Use the annotation tool to annotate the image with bounding boxes, click Complete when you are done"
    )
    tab1, tab2 = st.tabs(["Annotate", "Load mask"])
    with tab1:
        cols = st.columns(3)
        with cols[0]:
            selected_class = st.selectbox(
                "Select the class you want to annotate",
                st.session_state[SS.CLASSES],
                key="selectbox_class",
            )
            prompt_type = "polygon"
        with cols[1]:
            edit_mode = st.checkbox("Edit annotations", key="edit_mode")
        with cols[2]:
            focused = st.checkbox("Focused")
        edit_mode = "transform" if edit_mode else prompt_type
        selected_class_color_f, selected_class_color_st, selected_class_color_focused = get_color_from_class(
            st.session_state[SS.CLASSES], selected_class
        )
        shape = get_preprocess_shape(
            support_image.size[1], support_image.size[0], PROMPT_SIZE
        )
        selected_class_color_f = selected_class_color_focused if focused else selected_class_color_f
        selected_class_color_st = "white" if focused else selected_class_color_st
        results = st_canvas(
            fill_color=selected_class_color_f,  # Fixed fill color with some opacity
            stroke_color=selected_class_color_st,  # Fixed stroke color with full opacity
            background_image=support_image,
            drawing_mode=edit_mode,
            key="input_prompt_detection",
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
                    key=f"color_{cls}",
                    index=i,
                )
                color_map[i] = np.array(COLORS[TEXT_COLORS.index(color)])
        mask = st.file_uploader(
            "Upload the mask",
            type=["png", "jpg"],
            accept_multiple_files=False,
            key="mask_support",
        )
        mask = np.array(open_rgb_image(mask)) if mask is not None else None
        (
            st.image(mask, caption="Mask", use_column_width=True)
            if mask is not None
            else None
        )
        if mask is not None:
            results = {
                "mask": mask,
                "color_map": color_map,
            }
    if results is not None and st.button("Add Support Image"):
        add_annotated_support_image(support_image, results, shape)


def add_annotated_support_image(support_image, results, shape):
    example = SupportExample(support_image=support_image)
    if hasattr(results, "json_data") and results.json_data is not None:
        st.write("Extracting prompts from canvas")
        example.prompts = results.json_data
        example.reshape = shape
    if isinstance(results, dict) and "mask" in results:
        st.write("Extracting prompts from mask")
        example.prompts = results
        example.reshape = shape
    st.session_state[SS.SUPPORT_SET].append(example)
    st.session_state.pop("input_prompt_detection", None)
    st.session_state.pop("mask_support", None)
    st.session_state.pop("support_image", None)
    st.write("Support image added")


def preview_support_set(batch, preview_cols):
    for i, elem in enumerate(st.session_state[SS.SUPPORT_SET]):
        img = batch[BatchKeys.IMAGES][0][i]
        masks = batch[BatchKeys.PROMPT_MASKS][0][i]
        img = get_image(img)
        img = draw_masks(img, masks=masks, colors=COLORS)
        with preview_cols[i]:
            if st.button(f"Remove Image {i+1}"):
                reset_support(i)
            st.image(img, caption=f"Support Image {i+1}", use_column_width=True)
            
            
def rect_explanation(query_image_pt, attns_class, masks, flag_examples, num_sample):
    st.write("## Rectangle Explanation")
    
    masks = masks[:, :, 1:, ::]
    shape = query_image_pt.shape[-2:]
    query_image_pt = denormalize(query_image_pt)
    query_image_pt = query_image_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    query_image_pt = Image.fromarray((query_image_pt * 255).astype(np.uint8))

    results = st_canvas(
        fill_color="white",  # Fixed fill color with some opacity
        stroke_color="black",  # Fixed stroke color with full opacity
        background_image=query_image_pt,
        drawing_mode="rect",
        key=f"explain_input_{num_sample}",
        width=shape[1],
        height=shape[0],
        stroke_width=2,
        update_streamlit=False,
    )
    if results is not None and results.json_data is not None:
        if rects := [
        (rect["left"], rect["top"], rect['width'], rect["height"]) for rect in results.json_data["objects"]
        ]:
            rect = rects[0]
            rect = torch.tensor(rect)

            target_size = 48
            st.write(rect)

            for j, attns in enumerate(attns_class):                    
                attns = [
                    attn.mean(dim=1) for attn in attns
                ]
                class_examples = flag_examples[:, :, j + 1]
                mask = masks[:, :, j, ::][class_examples]

                rect_attns = []
                for attn in attns:
                    hw = attn.shape[-1]
                    h = w = int(hw ** 0.5)
                    scaled_rect = (rect * h / shape[0]).int()
                    mask_current = resize(mask, (h, w), interpolation=TvT.InterpolationMode.NEAREST)
                    mask_current = rearrange(mask_current, "1 h w -> 1 1 (h w)")
                    attn = attn * mask_current
                    attn = rearrange(attn, "b (h1 w1) (h2 w2) -> b h1 w1 h2 w2", h1=h, w1=w, h2=h, w2=w)
                    x0 = scaled_rect[0]
                    x1 = scaled_rect[0] + max(scaled_rect[2], 1)
                    y0 = scaled_rect[1]
                    y1 = scaled_rect[1] + max(scaled_rect[3], 1)
                    rect_attn = attn[0, x0:x1, y0:y1]
                    rect_attn = rect_attn.mean(dim=(0, 1))
                    rect_attn_norm = (rect_attn - rect_attn.min()) / (rect_attn.max() - rect_attn.min())
                    rect_attns.append((rect_attn, rect_attn_norm))
                st.write(f"Attention for class {j}")
                
                st.write(f"Attention summary for rect")
                mean_attn = torch.cat([
                    resize(attn[0].unsqueeze(0), (target_size, target_size)) for attn in rect_attns
                ]).mean(dim=0)
                mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min())
                st.write(mean_attn.chans.fig)
                
                st.write("Each channel attention")
                n_cols = 4
                cols = st.columns(n_cols)
                for k, (_, attn) in enumerate(rect_attns):
                    n_col = k % n_cols
                    cols[n_col].write(attn.chans.fig)
                    
def attention_summary(attns_class, masks, flag_examples):
    
    masks = masks[:, :, 1:, ::]
    st.write("## Attention Summary")
    target_size = 48
    for j, attns in enumerate(attns_class):
        attns = [
            attn.mean(dim=1) for attn in attns
        ]
        class_examples = flag_examples[:, :, j + 1]
        mask = masks[:, :, j, ::][class_examples]
        outs = []
        for attn in attns:
            hw = attn.shape[-1]
            h = w = int(hw ** 0.5)
            # resize mask to attn
            mask = resize(mask, (h, w), interpolation=TvT.InterpolationMode.NEAREST)
            mask = rearrange(mask, "1 h w -> 1 1 (h w)")
            attn = attn * mask
            attn = attn.sum(dim=-1)
            # attn = torch.matmul(attn, mask)
            attn = rearrange(attn, "1 (h w) -> 1 h w", h=h, w=w)
            attn = resize(attn, (target_size, target_size))
            outs.append(attn)
        out = torch.cat(outs).mean(dim=0)
        out = (out - out.min()) / (out.max() - out.min())
        st.write(out)
        st.write(out.chans.fig)

def try_it_yourself(model):
    st.write("Upload the image the you want to segment")
    query_images = st.file_uploader(
        "Choose an image", type=["png", "jpg", "webp"], accept_multiple_files=True
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
    if SS.SUPPORT_SET in st.session_state and len(st.session_state[SS.SUPPORT_SET]) > 0:
        preview_cols = st.columns((len(st.session_state[SS.SUPPORT_SET])))
        focusing_factor = st.number_input("Focusing Factor", min_value=1, max_value=100, value=5)
        support_batch = preprocess_support_set(
            st.session_state[SS.SUPPORT_SET],
            list(range(len(st.session_state[SS.CLASSES]))),
            preprocess=preprocess,
            device=st.session_state.get("device", "cpu"),
            custom_preprocess=CUSTOM_PREPROCESS,
            focusing_factor=focusing_factor
        )
        preview_support_set(support_batch, preview_cols)

    if (
        SS.SUPPORT_SET in st.session_state
        and len(st.session_state[SS.SUPPORT_SET]) > 0
        and SS.CLASSES in st.session_state
        and len(query_images) > 0
    ):
        batches = [
            preprocess_to_batch(
                image,
                support_batch.copy(),
                preprocess,
                device=st.session_state.get("device", "cpu"),
            )
            for image in images
        ]
        st.write(batches)
        st.write("## Predictions")
        if st.button("Predict"):
            st.session_state[SS.RESULT] = []
            progress = st.progress(0)
            for support_batch in batches:
                result = predict(model, support_batch)
                st.session_state[SS.RESULT].append(result)
                progress.progress((i + 1) / len(batches))
        if SS.RESULT in st.session_state:
            tabs = st.tabs([f"Query Image {i+1}" for i in range(len(batches))])
            for i, (support_batch, result) in enumerate(
                zip(batches, st.session_state[SS.RESULT])
            ):
                with tabs[i]:
                    pred = result[ResultDict.LOGITS].argmax(dim=1)
                    st.json(result, expanded=False)
                    plots, titles = plot_seg(
                        open_rgb_image(query_image),
                        pred,
                        COLORS,
                        dims=support_batch[BatchKeys.DIMS],
                        classes=st.session_state[SS.CLASSES],
                    )
                    cols = st.columns(2)
                    cols[0].image(
                        plots[0], caption=titles[0], use_column_width=True
                    )
                    cols[1].image(
                        plots[1], caption=titles[1], use_column_width=True
                    )
                attns_class = result.get(ResultDict.ATTENTIONS, None)
                if attns_class is not None:
                    query_image_pt = batches[i][BatchKeys.IMAGES][0, 0]
                    rect_explanation(query_image_pt, attns_class, batches[i][BatchKeys.PROMPT_MASKS], batches[i][BatchKeys.FLAG_EXAMPLES], i)
                    attention_summary(attns_class, batches[i][BatchKeys.PROMPT_MASKS], batches[i][BatchKeys.FLAG_EXAMPLES])


def handle_gpu_memory(device):
    # Display GPU memory
    if device == "cuda":
        allocated = f"{torch.cuda.memory_allocated() / 1e9:.2f}"
        total = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}"
        st.progress(
            torch.cuda.memory_allocated()
            / torch.cuda.get_device_properties(0).total_memory,
            text=f"GPU memory allocated: {allocated} GB / {total} GB",
        )
    if st.button("Clear GPU cache") and device == "cuda":
        torch.cuda.empty_cache()


def main():
    st.set_page_config(layout="wide", page_title="Focused FSS")
    st.title("Focused FSS")
    st.sidebar.title("Settings")
    with st.sidebar:
        if cuda := torch.cuda.is_available():
            use_gpu = st.checkbox("Use GPU", True)
        device = "cuda" if cuda and use_gpu else "cpu"
        st.session_state["device"] = device
        # load model
        st.write("Working on device:", device)
        model = load_model(device)
        
        st.divider()
        st.json(st.session_state, expanded=False)
        handle_gpu_memory(device)
    tiy_tab, dataset_tab = st.tabs(["Try it yourself", "Built-in dataset"])
    with tiy_tab:
        try_it_yourself(model)
    with dataset_tab:
        built_in_dataset(model)


if __name__ == "__main__":
    main()
