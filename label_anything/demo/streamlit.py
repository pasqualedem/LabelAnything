import itertools
from einops import rearrange
import pandas as pd
import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from streamlit_image_annotation import detection
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
from label_anything.experiment.substitution import Substitutor
from label_anything.models.build_vit import build_vit_b
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
    plot_seg_gt,
    plot_seg,
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


IMG_DIR = "/ext/stalla/LabelAnything/images/train2017"
ANNOTATIONS_DIR = "data/annotations/instances_val2017.json"
EMBEDDINGS_DIR = "/ext/stalla/LabelAnything/embeddings"
MAX_EXAMPLES = 30
VIT_B_SAM_PATH = "checkpoints/sam_vit_b_01ec64.pth"

SIZE = 1024

preprocess = Compose([CustomResize(SIZE), PILToTensor(), CustomNormalize(SIZE)])


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
    folder = "latest"
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

    cat_ids = sorted(list(set(itertools.chain(*aux_cat_ids))))
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


@st.cache_resource(hash_funcs={torch.Tensor: lambda x: x.sum().item()})
def get_features(_model, batch):
    b, n = batch.shape[:2]
    batch = rearrange(batch, "b n c h w -> (b n) c h w")
    with torch.no_grad():
        result = torch.cat([_model(batch[i].unsqueeze(0)) for i in range(batch.shape[0])], dim=0)
    result = rearrange(result, "(b n) c h w -> b n c h w", b=b)
    return result


def predict(model, image_encoder, batch):
    image_features = get_features(image_encoder, batch[BatchKeys.IMAGES])
    batch[BatchKeys.EMBEDDINGS] = image_features
    with torch.no_grad():
        result = model(batch)
    return result


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
    plots, titles = plot_seg_gt(
        input,
        seg,
        one_gt,
        colors,
        dims=input[BatchKeys.DIMS],
        classes=input["classes"][0][0],
    )
    cols = st.columns(2)
    for i, plot in enumerate(plots):
        cols[i % 2].image(plot, caption=titles[i], use_column_width=True)


def built_in_dataset(accelerator, model):
    st.write("## Under maintanaince")
    return
    st.text_input("Image directory", IMG_DIR, key="img_dir")
    st.text_input("Annotations directory", ANNOTATIONS_DIR, key="annotations_dir")

    datalaoder = get_data(accelerator)
    st.session_state["dataset"] = (
        get_data(_accelerator=accelerator) if st.button("Load dataset") else None
    )
    dataset = st.session_state.get("dataset", None)
    if dataset is None:
        return
    coco = dataset.dataset.datasets["coco"]

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


def reset_support(idx, image=False):
    if idx is None:
        st.session_state["support_set"] = {}
        return
    if idx in st.session_state["support_set"]:
        st.session_state["support_set"][idx]["prompts"]["bboxes"] = []
        st.session_state["support_set"][idx]["prompts"]["labels"] = []
        if image:
            st.session_state["support_set"].pop(idx)


def build_support_set():
    st.write("Choose the classes you want to segment in the image")
    classes = st_tags(
        label="Classes",
        text="Type and press enter",
        value=[],
        suggestions=["person", "car", "dog", "cat", "bus", "truck"],
    )
    if len(classes) < len(st.session_state.get("classes", [])):  # Reset annotations
        for k in st.session_state["support_set"].keys():
            reset_support(k)
    if not classes:
        return
    st.write(classes)
    st.session_state["classes"] = classes
    if st.session_state.get("support_set", None) is None:
        st.session_state["support_set"] = {}
    st.write("## Upload and annotate the support images")
    i = 0
    support_image = st.file_uploader(
        f"Choose the first support image",
        type=["png", "jpg"],
        key=f"support_image_{i}",
        on_change=lambda: reset_support(None, True),
    )
    while support_image is not None:
        add_support_image(support_image, i)
        i += 1
        support_image = st.file_uploader(
            f"If you want, you can upload and annotate another support image",
            type=["png", "jpg"],
            on_change=lambda: reset_support(None, True),
            key=f"support_image_{i}",
        )
    st.write("Review the support set")
    st.json(st.session_state["support_set"], expanded=False)


def add_support_image(support_image, idx):
    support_image = Image.open(support_image)
    # Save image in a temp file
    tmp_support_file = NamedTemporaryFile(delete=False)
    support_image.save(tmp_support_file.name + ".png")
    if idx not in st.session_state["support_set"]:
        st.session_state["support_set"][idx] = {}
        st.session_state["support_set"][idx]["img"] = support_image
    if "prompts" not in st.session_state["support_set"][idx]:
        st.session_state["support_set"][idx]["prompts"] = {}
        st.session_state["support_set"][idx]["prompts"]["bboxes"] = []
        st.session_state["support_set"][idx]["prompts"]["labels"] = []
    st.write(
        f"Use the annotation tool to annotate the image with bounding boxes, click Complete when you are done"
    )
    results = detection(
        tmp_support_file.name + ".png",
        label_list=st.session_state["classes"],
        labels=st.session_state["support_set"][idx]["prompts"]["labels"],
        height=support_image.height,
        width=support_image.width,
        bboxes=st.session_state["support_set"][idx]["prompts"]["bboxes"],
        key=f"input_prompt_detection_{idx}",
    )
    if results is not None:
        st.session_state["support_set"][idx]["prompts"]["bboxes"] = [
            v["bbox"] for v in results
        ]
        st.session_state["support_set"][idx]["prompts"]["labels"] = [
            v["label_id"] for v in results
        ]


def try_it_yourself(model, image_encoder):
    st.write("Upload the image the you want to segment")
    query_image = st.file_uploader("Choose an image", type=["png", "jpg"])
    image_placeholder = st.empty()
    if query_image is not None:
        image = Image.open(query_image)
        # Save image in a temp file
        image_placeholder.image(image, caption="Query Image", width=300)
        build_support_set()
        if "support_set" not in st.session_state:
            return
        batch = preprocess_to_batch(
            image,
            st.session_state["support_set"],
            list(range(len(st.session_state["classes"]))),
        )
        if st.button("Predict"):
            result = predict(model, image_encoder, batch)
            st.json(result, expanded=False)
            plots, titles = plot_seg(
                batch,
                result[ResultDict.LOGITS].argmax(dim=1),
                colors,
                dims=batch[BatchKeys.DIMS],
                classes=st.session_state["classes"],
            )
            cols = st.columns(2)
            cols[0].image(plots[0], caption=titles[0], use_column_width=True)
            cols[1].image(plots[1], caption=titles[1], use_column_width=True)


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
    tiy_tab, dataset_tab = st.tabs(["Try it yourself", "Built-in dataset"])
    with tiy_tab:
        try_it_yourself(model, image_encoder)
    with dataset_tab:
        built_in_dataset(accelerator, model)


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
