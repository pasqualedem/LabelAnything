import itertools
from einops import rearrange
import pandas as pd
import streamlit as st
import torch
import numpy as np
import plotly.express as px
import torch



from label_anything.demo.utils import COLORS, TEXT_COLORS, get_data
from label_anything.experiment.substitution import Substitutor
from label_anything.utils.utils import ResultDict
from label_anything.data.utils import (
    AnnFileKeys,
    PromptType,
    BatchKeys,
)
from label_anything.experiment.utils import WrapperModule
import zipfile
import os
from io import BytesIO

from label_anything.demo.visualize import (
    get_embeddings_names,
    obtain_batch,
    plot_seg_gt,
    draw_all,
    get_image,
    reduce_embeddings,
    set_embeddings,
    to_device,
)


def get_prompt_types():
    options = {PromptType.BBOX, PromptType.MASK, PromptType.POINT}
    cols = st.columns(len(options))
    toggles = {
        option: cols[i].checkbox(option.value, value=True)
        for i, option in enumerate(options)
    }
    return [option for option, toggle in toggles.items() if toggle]


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
            COLORS,
            unbatched["flag_masks"][i],
            unbatched["flag_bboxes"][i],
            unbatched["flag_points"][i],
        )
        for i in range(unbatched["images"].shape[0])
    ]
    max_cols = 4
    n_cols = min(max_cols, len(images) - 1)
    cols = st.columns(n_cols)
    for i, image in enumerate(images[1:]):
        cols[i % 4].image(image, caption=f"Prompt {i+1}", use_column_width=True)
    return images


def get_result(model, batch, gt):
    device = st.session_state.get("device", "cpu")
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    substitutor = Substitutor(substitute=False)
    substitutor.reset((batch, gt))
    input, one_gt = next(iter(substitutor))
    with torch.no_grad():
        result = model(input)
    st.session_state["input"] = input
    st.session_state["result"] = result
    st.session_state["one_gt"] = one_gt
    st.session_state["seg"] = result[ResultDict.LOGITS].argmax(dim=1)


@st.cache_resource(hash_funcs={torch.Tensor: lambda x: x.sum().item()})
def get_features(_model, batch):
    b, n = batch.shape[:2]
    batch = rearrange(batch, "b n c h w -> (b n) c h w")
    with torch.no_grad():
        result = torch.cat(
            [_model(batch[i].unsqueeze(0)) for i in range(batch.shape[0])], dim=0
        )
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
    plot_colors = ["white"] + TEXT_COLORS
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
    # if result is not None:
    #     plot_embeddings(
    #         examples_class_embeddings=result[ResultDict.EXAMPLES_CLASS_EMBS],
    #         example_flags=input[BatchKeys.FLAG_EXAMPLES],
    #     )
    plots, titles = plot_seg_gt(
        input,
        seg,
        one_gt,
        COLORS,
        dims=input[BatchKeys.DIMS],
        classes=input["classes"][0][0],
    )
    n_cols = 5
    cols = st.columns(n_cols)
    for i, (title, plot) in enumerate(zip(titles, plots)):
        cols[i % n_cols].write(title)
        cols[i % n_cols].write(plot.rgb.fig)
    return plots, titles


def built_in_dataset(model):
    IMAGE_SIZE = 480

    with st.expander("Dataset parameters", expanded=True):
        col1, col2 = st.columns(2)
        n_ways = col1.number_input("Number of ways", 1, 20, 1)
        n_shots = col2.number_input("Number of shots", 1, 20, 1)
        n_examples = None
        if col2.checkbox("Use shots as examples", value=False):
            n_examples = n_shots
            n_shots = None
        custom_preprocess = col1.checkbox("Segment Anytihng preprocess", value=False)
        image_size = col2.number_input(
            "Image size", 1, 1024, IMAGE_SIZE, step=32
        )
        prompt_types = get_prompt_types()
        max_points = col1.number_input(
            "Max points per annotation", 1, 20, 5
        )
        all_example_categories = col1.checkbox(
            "Use all categories in examples", value=False
        )
        class_based_sampling = col1.checkbox(
            "Class based sampling", value=True
        )
        fold = col2.number_input(
            "Fold", 0, 3, 0
        )

    if st.button("Load dataset"):
        dataloader = get_data(n_ways, n_shots, n_examples, image_size, custom_preprocess, all_example_categories, prompt_types, max_points, fold, class_based_sampling=True)
        st.session_state["dataset"] = iter(dataloader)
        st.session_state["datasets"] = dataloader.dataset.datasets
    dataset = st.session_state.get("dataset", None)
    if dataset is None:
        return

    if st.button("Get sample"):
        (batch, gt), dataset_name = next(dataset)
        st.session_state["batch"] = (batch, gt, dataset_name[0])
    if st.session_state.get("batch", None) is None:
        return

    batch, gt, dataset_name = st.session_state.get("batch", (None, None, None))
    st.write(batch)

    images = batch.get("images", [])[0]
    cat_ids = batch.get("classes", [])[0]
    cols = st.columns(len(images))
    for i, image in enumerate(images):
        image = get_image(image)
        cats = [st.session_state["datasets"][dataset_name].categories[cat]["name"] for cat in cat_ids[i]]
        cols[i].image(image, caption=f"{cats}", use_column_width=True)
    st.session_state["plot_prompts"] = plot_prompts()
    if st.button("Predict"):
        get_result(model, batch, gt)
    if st.session_state.get("result", None) is not None:
        st.session_state["plot_results"] = plot_results()
    if "plot_prompts" not in st.session_state or  "plot_results" not in st.session_state:
        return
    if st.button("Download Results"):
        # Create a temporary zip file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Save prompt plots
            for i, image in enumerate(st.session_state["plot_prompts"]):
                image_path = f"prompt_{i + 1}.png"
                image.save(image_path)
                zip_file.write(image_path, arcname=os.path.basename(image_path))
                os.remove(image_path)

            # Save result plots
            for i, (plot, title) in enumerate(
                zip(st.session_state["plot_results"][0], st.session_state["plot_results"][1])
            ):
                result_path = f"result_{i + 1}_{title}.png"
                plot.rgb.fig.savefig(result_path)
                zip_file.write(result_path, arcname=os.path.basename(result_path))
                os.remove(result_path)

        # Prepare the zip file for download
        zip_buffer.seek(0)
        st.download_button(
        label="Download ZIP",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip",
        )