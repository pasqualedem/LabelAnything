import itertools
import pandas as pd
import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


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
from label_anything.experiment.substitution import Substitutor
from label_anything.utils.utils import ResultDict, load_yaml
from label_anything.models import model_registry

lt.monkey_patch()

from label_anything.models import build_lam_no_vit
from label_anything.data.examples import uniform_sampling
from label_anything.data import utils
from label_anything.data.utils import (
    AnnFileKeys,
    PromptType,
    BatchKeys,
    get_preprocess_shape,
)
from label_anything.experiment.utils import WrapperModule

from label_anything.demo.visualize import (
    get_embeddings_names,
    load_from_wandb,
    obtain_batch,
    plot_emebddings,
    # plot_segs,
    draw_all,
    get_image,
    colors,
    reduce_embeddings,
    set_embeddings,
    text_colors,
    to_device,
)

import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize


def draw_seg(img: Image, seg: torch.Tensor, colors, num_classes, dim=None):
    print("----")
    print("dim", dim)
    print("img size", img.size)
    print("seg shape", seg.shape)
    h, w = get_preprocess_shape(int(dim[0]), int(dim[1]), 1024)
    img = img.crop((0, 0, w, h))
    print("cropped size", img.size)
    resized_image = resize(img.copy(), (seg.shape[-2], seg.shape[-1]))
    print("resized ", resized_image.size)
    masked_image = resized_image.copy()
    for i in range(1, num_classes):
        binary_mask = (seg == i)[0]
        mask = binary_mask.cpu().numpy()
        masked_image = np.where(
            np.repeat(mask[:, :, np.newaxis], 3, axis=2),
            np.asarray(colors[i], dtype="uint8"),
            masked_image,
        )

    masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(np.array(resized_image), 0.6, masked_image, 0.4, 0)


def plot_segs(input, seg, gt, colors, dims):
    query_dim = dims[0, 0]
    num_classes = len(input["classes"][0][0]) + 1
    image = get_image(input["images"][0, 0])
    segmask = draw_seg(image, seg.cpu(), colors, num_classes=num_classes, dim=query_dim)
    gtmask = draw_seg(image, gt, colors, num_classes=num_classes, dim=query_dim)
    blank_seg = Image.fromarray(np.zeros_like(segmask))
    blank_gt = Image.fromarray(np.zeros_like(gtmask))
    blank_segmask = draw_seg(
        blank_seg, seg.cpu(), colors, num_classes=num_classes, dim=query_dim
    )
    blank_gtmask = draw_seg(
        blank_gt, gt, colors, num_classes=num_classes, dim=query_dim
    )
    plots = [segmask, gtmask, blank_segmask, blank_gtmask, image, image]
    titles = [
        "Predicted",
        "Ground Truth",
        "Predicted",
        "Ground Truth",
        "Original",
        "Original",
    ]

    subplots = plt.subplots(3, 2, figsize=(20, 30))
    for i, (plot, title) in enumerate(zip(plots, titles)):
        subplots[1].flatten()[i].imshow(plot)
        subplots[1].flatten()[i].set_title(title)
        subplots[1].flatten()[i].axis("off")
    return plots, titles


IMG_DIR = "/ext/stalla/LabelAnything/images/train2017"
ANNOTATIONS_DIR = "data/annotations/instances_val2017.json"
EMBEDDINGS_DIR = "/ext/stalla/LabelAnything/embeddings"
MAX_EXAMPLES = 30

SIZE = 1024

preprocess = Compose([CustomResize(SIZE), PILToTensor(), CustomNormalize(SIZE)])


@st.cache_resource
def get_data(_accelerator):
    dataset = LabelAnythingDataset(
        {
            "coco": {
                "name": "coco",
                "instances_path": ANNOTATIONS_DIR,
                "img_dir": IMG_DIR,
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
    folder = "latest"
    model_file, config_file = load_from_wandb(run_id, folder)
    if config_file is not None:
        config = load_yaml(config_file)
        model_params = config["model"]["value"]
        name = model_params.pop("name")
    else:
        model_params = {}
        name = "lam_no_vit"
        st.warning(
            f"Config file not found, using default model params: {model_params}, {name}"
        )
    model = model_registry[name](**model_params)
    model = WrapperModule(model, None)
    model_state_dict = torch.load(model_file)
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


def show_image(idx, coco_dataset):
    base_image_data = coco_dataset.images[coco_dataset.image_ids[idx]]

    st.image(get_image(coco_dataset._load_and_preprocess_image(base_image_data)))

    img_cats = torch.tensor(list(coco_dataset.img2cat[base_image_data[AnnFileKeys.ID]]))
    sampled_classes = (
        coco_dataset.example_generator.sample_classes_from_query(
            img_cats, uniform_sampling
        )
        if coco_dataset.do_subsample
        else img_cats
    )
    cols = st.columns(len(img_cats))
    for i, cat in enumerate(img_cats):
        cols[i].checkbox(
            coco_dataset.categories[cat.item()]["name"],
            value=True,
            key=f"cat_{cat}",
        )
    chosen_classes = [
        cat.item() for i, cat in enumerate(img_cats) if st.session_state[f"cat_{cat}"]
    ]
    st.write(
        f"Chosen classes: {[coco_dataset.categories[cat]['name'] for cat in chosen_classes]}"
    )
    if "chosen_classes" not in st.session_state:
        st.session_state["chosen_classes"] = chosen_classes
    elif st.session_state["chosen_classes"] != chosen_classes:
        st.session_state["chosen_classes"] = chosen_classes
        st.session_state["batch"] = (None, None, None)
        st.session_state["result"] = None
        st.session_state["examples"] = None

    return base_image_data, img_cats, chosen_classes


def generate_examples(
    coco_dataset, base_image_data, img_cats, chosen_classes, num_examples
):
    image_ids, aux_cat_ids = coco_dataset.example_generator.generate_examples(
        query_image_id=base_image_data[AnnFileKeys.ID],
        image_classes=img_cats,
        sampled_classes=torch.tensor(chosen_classes),
        num_examples=num_examples,
    )

    cat_ids = list(set(itertools.chain(*aux_cat_ids)))
    st.session_state["named_classes"] = ["background"] + [
        coco_dataset.categories[cat]["name"] for cat in cat_ids
    ]
    cat_ids.insert(0, -1)  # add the background class

    images, image_key, ground_truths = coco_dataset._get_images_or_embeddings(image_ids)
    pil_images = [get_image(image) for image in images][1:]
    st.session_state["examples"] = {
        "images": images,
        "pil_images": pil_images,
        "aux_cat_ids": aux_cat_ids,
        "image_ids": image_ids,
        "cat_ids": cat_ids,
        "image_key": image_key,
        "ground_truths": ground_truths,
    }
    st.session_state["result"] = None
    st.session_state["batch"] = (None, None, None)


def get_prompt_types():
    options = {PromptType.BBOX, PromptType.MASK, PromptType.POINT}
    cols = st.columns(len(options))
    toggles = {
        option: cols[i].checkbox(option.value, value=True)
        for i, option in enumerate(options)
    }
    return [option for option, toggle in toggles.items() if toggle]


def generate_prompts(accelerator, dataset, coco_dataset, prompt_types):
    image_ids = st.session_state.get("examples", {}).get("image_ids", [])
    cat_ids = st.session_state.get("examples", {}).get("cat_ids", [])
    images = st.session_state.get("examples", {}).get("images", [])
    st.write(f"Image ids: {image_ids}")
    st.write(f"Cat ids: {cat_ids}")
    bboxes, masks, points, classes, img_sizes = coco_dataset._get_prompts(
        image_ids, cat_ids, prompt_types
    )
    image_key = st.session_state.get("examples", {}).get("image_key", [])
    (batch, gt), dataset_name = obtain_batch(
        dataset,
        coco_dataset,
        images,
        image_ids,
        cat_ids,
        classes,
        img_sizes,
        image_key,
        (bboxes, masks, points),
        ground_truths=None,
    )
    batch = to_device(batch, accelerator.device)
    st.session_state["batch"] = (batch, gt, dataset_name)
    st.session_state["result"] = None


def plot_prompts():
    batch, gt, dataset_name = st.session_state.get("batch", (None, None, None))
    if batch is None:
        return
    unbatched = {k: v[0] for k, v in batch.items()}
    images = [
        draw_all(
            get_image(unbatched["images"][i]),
            unbatched["prompt_masks"][i],
            unbatched["prompt_bboxes"][i],
            unbatched["prompt_points"][i],
            colors,
        )
        for i in range(unbatched["images"].shape[0])
    ]
    max_cols = 4
    n_cols = min(max_cols, len(images) - 1)
    cols = st.columns(n_cols)
    for i, image in enumerate(images[1:]):
        cols[i % 4].image(image, caption=f"Prompt {i+1}", use_column_width=True)


def get_result(model, batch, gt):
    substitutor = Substitutor(substitute=False)
    substitutor.reset((batch, gt))
    input, one_gt = next(iter(substitutor))
    with torch.no_grad():
        result = model(input)
    st.session_state["input"] = input
    st.session_state["result"] = result
    st.session_state["one_gt"] = one_gt
    st.session_state["seg"] = result[ResultDict.LOGITS].argmax(dim=1)
    st.session_state["reduced_embeddings"] = reduce_embeddings(
        result[ResultDict.EXAMPLES_CLASS_EMBS]
    )


def plot_embeddings(examples_class_embeddings, example_flags):
    embeddings_2d = st.session_state.get("reduced_embeddings", None)
    b, n, c, _ = examples_class_embeddings.shape
    n_example = np.repeat(np.arange(c), repeats=n)
    n_class = np.tile(np.arange(c), n)
    class_names = st.session_state["named_classes"]
    n_class = [class_names[i] for i in n_class]
    st.write(f"Class names: {class_names}")
    valid = example_flags.flatten().cpu().numpy() * 10 + 1
    df = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "example": n_example,
            "class": n_class,
            "valid": valid,
        }
    )
    df["class"] = df["class"].astype("str")
    plot_colors = ["white"] + text_colors
    scatter_plot = px.scatter(
        df,
        x="x",
        y="y",
        color="class",
        symbol="example",
        size="valid",
        color_discrete_sequence=plot_colors,
    )

    st.plotly_chart(scatter_plot)


def plot_results():
    input = st.session_state.get("input", None)
    result = st.session_state.get("result", None)
    one_gt = st.session_state.get("one_gt", None)
    seg = st.session_state.get("seg", None)
    if result is not None:
        plot_embeddings(
            examples_class_embeddings=result[ResultDict.EXAMPLES_CLASS_EMBS],
            example_flags=input[BatchKeys.FLAG_EXAMPLES],
        )
    plots, titles = plot_segs(input, seg, one_gt, colors, dims=input[BatchKeys.DIMS])
    cols = st.columns(2)
    for i, plot in enumerate(plots):
        cols[i % 2].image(plot, caption=titles[i], use_column_width=True)


def main():
    st.set_page_config(layout="wide", page_title="Label Anything")
    st.title("Label Anything")
    st.sidebar.title("Settings")
    accelerator = Accelerator()
    # uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg"])
    dataset = get_data(_accelerator=accelerator)
    coco = dataset.dataset.datasets["coco"]
    with st.sidebar:
        # load model
        run_id = st.text_input("Run ID", "4pdvxpgy")
        model = load_model(accelerator, run_id).model  # WrapperModule
        datalaoder = get_data(accelerator)
        image_idx = st.slider("Image index", 0, len(dataset) - 1, 0)
        num_examples = st.slider("Number of examples", 1, MAX_EXAMPLES, 1)
        prompt_types = get_prompt_types()
    base_image_data, img_cats, chosen_classes = show_image(image_idx, coco)
    if st.button("Generate Examples"):
        generate_examples(coco, base_image_data, img_cats, chosen_classes, num_examples)
    if st.session_state.get("examples", None) is not None:
        pil_images = st.session_state.get("examples", {}).get("pil_images", [])
        aux_cat_ids = st.session_state.get("examples", {}).get("aux_cat_ids", [])
        if pil_images:
            cols = st.columns(len(pil_images))
            for i, pil_image in enumerate(pil_images):
                cats = [coco.categories[cat]["name"] for cat in aux_cat_ids[i]]
                cols[i].image(pil_image, caption=f"{cats}", use_column_width=True)
        if st.button("Generate prompts"):
            generate_prompts(accelerator, datalaoder.dataset, coco, prompt_types)
        plot_prompts()
        batch, gt, dataset_name = st.session_state.get("batch", (None, None, None))
        if batch is not None:
            embeddings_name = get_embeddings_names(batch, EMBEDDINGS_DIR)
            st.write(f"Needed embeddings")
            st.code(embeddings_name)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load embeddings"):
                    batch = set_embeddings(accelerator, batch, EMBEDDINGS_DIR)
                    batch = to_device(batch, accelerator.device)
            with col2:
                if st.button("Predict"):
                    get_result(model, batch, gt)
            if st.session_state.get("result", None) is not None:
                plot_results()

    # load image
    # image_placeholder = row[0].empty()
    # if uploaded_image is not None:
    #     image = np.array(Image.open(uploaded_image))
    #     image_placeholder.image(image, caption="Uploaded Image", use_column_width=True)

    # show prediction
    # if uploaded_file is not None and uploaded_image is not None:
    #     button_row = st.columns([1, 1, 1])
    #     if button_row[1].button("Predict"):
    #         prediction = model(image)
    #         row[1].write("Prediction")
    #         row[1].image(prediction, caption="Prediction Image", use_column_width=True)


if __name__ == "__main__":
    main()


# @st.cache(allow_output_mutation=True)
# def load_data(name, instances_path):
#     return CocoLVISTestDataset(
#         name,
#         instances_path,
#         max_num_examples=10,
#         preprocess=preprocess,
#     )


# def show_batch(images: torch.Tensor):
#     for i, image in enumerate(images):
#         image_array = image.permute(1, 2, 0).numpy()
#         image_array = (image_array - np.min(image_array)) / (
#             np.max(image_array) - np.min(image_array)
#         )
#         st.write(f"### Source image {i+1}")
#         st.image(image_array, caption=f"Immagine {i+1}", use_column_width=True)


# def show_gt(gts):
#     for i, gt in enumerate(gts):
#         gt_array = gt.numpy()
#         gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))
#         st.write(f"### Ground Truth {i+1}")
#         st.image(gt_array, caption=f"Ground {i+1}")


# def main():
#     st.title("Label Anything")
#     dataset = load_data(name, source)
#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=2,
#         shuffle=False,
#         collate_fn=dataset.collate_fn,
#     )

#     dataloader_iter = iter(dataloader)

#     data_dict, gt = next(dataloader_iter)
#     images = data_dict["images"]
#     column1, column2 = st.columns(2)
#     show_batch(images)
#     show_gt(gt)
