import itertools
from einops import rearrange
import pandas as pd
import streamlit as st
import torch
import numpy as np
import plotly.express as px


import numpy as np
import torch
import numpy as np

from label_anything.demo.utils import COLORS, TEXT_COLORS
from label_anything.experiment.substitution import Substitutor
from label_anything.utils.utils import ResultDict
from label_anything.data.examples import uniform_sampling
from label_anything.data import utils
from label_anything.data.utils import (
    AnnFileKeys,
    PromptType,
    BatchKeys,
)
from label_anything.experiment.utils import WrapperModule

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
            COLORS,
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
        result = torch.cat(
            [_model(batch[i].unsqueeze(0)) for i in range(batch.shape[0])], dim=0
        )
    result = rearrange(result, "(b n) c h w -> b n c h w", b=b)
    return result


def predict(model, batch):
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
    if result is not None:
        plot_embeddings(
            examples_class_embeddings=result[ResultDict.EXAMPLES_CLASS_EMBS],
            example_flags=input[BatchKeys.FLAG_EXAMPLES],
        )
    plots, titles = plot_seg_gt(
        input,
        seg,
        one_gt,
        COLORS,
        dims=input[BatchKeys.DIMS],
        classes=input["classes"][0][0],
    )
    cols = st.columns(2)
    for i, plot in enumerate(plots):
        cols[i % 2].image(plot, caption=titles[i], use_column_width=True)


def built_in_dataset(model):
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