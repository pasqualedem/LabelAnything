from tempfile import TemporaryDirectory
from einops import rearrange
from nicegui import ui, events, observables, run, app
from pathlib import Path
from PIL import Image
import torch

from label_anything.data.utils import BatchKeys
from label_anything.demo.preprocess import preprocess_to_batch
from label_anything.demo.visualize import (
    draw_all,
    get_image,
    load_from_wandb,
    plot_seg,
)
from label_anything.demo.utils import COLORS, retrieve_models
from label_anything.demo.preprocess_ng import preprocess_support_set
from label_anything.models.build_lam import LabelAnything
from label_anything.utils.utils import ResultDict

IMAGE_SIZE = 1024
DEVICE = "cpu"  # Change to "cuda" if you have a GPU
TEXT_COLORS = [
    # rbg to hex
    f"rgba({c[0]}, {c[1]}, {c[2]}, 1)"
    for c in COLORS
]


def preview_support_set(batch):
    # Clear previous preview if it exists
    if "preview" in components:
        components["preview"].clear()

    with components["preview"]:
        # Scrollable image row
        with ui.row().classes("gap-4 overflow-x-auto whitespace-nowrap pb-2"):
            for i, _ in enumerate(state["support_set"]):
                img = batch[BatchKeys.IMAGES][0][i]
                masks = batch[BatchKeys.PROMPT_MASKS][0][i]
                bboxes = batch[BatchKeys.PROMPT_BBOXES][0][i]
                points = batch[BatchKeys.PROMPT_POINTS][0][i]

                img = get_image(img)
                img = draw_all(
                    img, masks=masks, boxes=bboxes, points=points, colors=COLORS
                )

                with ui.element("div").classes(
                    "bg-white rounded-xl shadow-md p-2 flex flex-col items-center min-w-[180px] max-w-[256px]"
                ):
                    ui.image(img).style("width: 100%; max-width: 512px;").classes(
                        "rounded"
                    )
                    ui.label(f"Support Image {i + 1}").classes(
                        "text-sm text-gray-600 mt-2"
                    )


def set_batch_support():
    state["batch_support"] = preprocess_support_set(
        state["support_set"].copy(),
        state["classes"].copy(),
        size=IMAGE_SIZE,
        custom_preprocess=True,
        device="cpu",
    )
    ui.notify(f'Preprocessed {state["support_set"]} support images', color="positive")
    preview_support_set(state["batch_support"])


UPLOAD_DIR = TemporaryDirectory(prefix="label_anything_uploads", delete=False)

state = {
    "query_image_path": None,
    "support_image_path": None,
    "image_widget": None,
    "classes": observables.ObservableList(),
    "current_class": None,
    "support_set": observables.ObservableList([], on_change=set_batch_support),
    "batch_support": None,
}

components = {}

AVAILABLE_COLORS = set(TEXT_COLORS.copy())
TAKEN_COLORS = {}


def set_current_class(value):
    if value in state["classes"]:
        state["current_class"] = value
        ui.notify(f"Current class set to: {value}")
    else:
        ui.notify(f'Class "{value}" not found in available classes', color="negative")


def add_class(label_input: ui.input, chips):
    color = AVAILABLE_COLORS.pop()
    TAKEN_COLORS[label_input.value] = color
    if not AVAILABLE_COLORS:
        ui.notify("No more colors available for new classes", color="negative")
        return
    with chips:
        chip = ui.chip(
            label_input.value,
            icon="label",
            color=color,
            removable=True,
            on_click=lambda: set_current_class(chip.text),
        )
    chip.on("remove", lambda: remove_class(chip, chips))
    state["classes"].append(label_input.value)
    label_input.value = ""


def remove_class(chip: ui.chip, chips):
    chips.remove(chip)
    ui.notify(f'Removed class: {chip.text} from {state["classes"]}')
    state["classes"].remove(chip.text)
    AVAILABLE_COLORS.add(TAKEN_COLORS.pop(chip.text))


def end_polyline():
    current_class = state.get("current_class")
    if not current_class:
        ui.notify("Please select a class first by clicking on it", color="negative")
        return
    if not state["polyline"]:
        ui.notify("No polyline started", color="negative")
        return

    last_line = f"<line x1=\"{state['polyline'][-2]}\" y1=\"{state['polyline'][-1]}\" x2=\"{state['polyline'][0]}\" y2=\"{state['polyline'][1]}\" stroke=\"{TAKEN_COLORS[current_class]}\" stroke-width=\"3\" />"
    state["svg_annotations"].content += last_line

    if current_class not in state["annotations"]["masks"]:
        state["annotations"]["masks"][current_class] = []
    state["annotations"]["masks"][current_class].append(state["polyline"])
    state["polyline"] = None


def mouse_handler(e: events.MouseEventArguments):
    selected_prompt = state["prompt"].value
    current_class = state.get("current_class")
    color = TAKEN_COLORS.get(current_class)
    if not color:
        ui.notify("Please select a class first by clicking on it", color="negative")
        return
    # ui.notify(
    #     f"State prompt: {state['prompt'].value}, Mouse event: {e.type}, Coordinates: ({e.image_x}, {e.image_y}, Class: {current_class}, Color: {color})"
    # )
    if e.shift or selected_prompt == "Rectangle":  # Add a rectangle
        if state["start_rect"]:
            # Finish rectangle and draw it
            x1, y1 = state["start_rect"]
            x2, y2 = e.image_x, e.image_y
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            rect_svg = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="3" />'
            state["svg_annotations"].content += rect_svg
            if current_class not in state["annotations"]["bboxes"]:
                state["annotations"]["bboxes"][current_class] = []
            state["annotations"]["bboxes"][current_class].append((x, y, x + w, y + h))
            ui.notify(
                f"Rectangle completed: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})"
            )
            state["start_rect"] = None
        else:
            state["start_rect"] = (e.image_x, e.image_y)
            ui.notify(f"Starting rectangle at ({e.image_x}, {e.image_y})")
    elif e.ctrl or selected_prompt == "Mask":  # Add a polyline
        if state["polyline"]:
            state[
                "svg_annotations"
            ].content += f'<line x1="{state["polyline"][-2]}" y1="{state["polyline"][-1]}" x2="{e.image_x}" y2="{e.image_y}" stroke="{color}" stroke-width="3" />'
            state["polyline"].extend([e.image_x, e.image_y])
            # Add the line to annotations
            # Notify the user
            ui.notify(f"Line drawn to ({e.image_x}, {e.image_y})")
        else:
            state["polyline"] = [e.image_x, e.image_y]
            ui.notify(f"Starting line at ({e.image_x}, {e.image_y})")
    else:
        state[
            "svg_annotations"
        ].content += f'<circle cx="{e.image_x}" cy="{e.image_y}" r="15" fill="none" stroke="{color}" stroke-width="4" />'
        if current_class not in state["annotations"]["points"]:
            state["annotations"]["points"][current_class] = []
        state["annotations"]["points"][current_class].append((e.image_x, e.image_y))
        ui.notify(f"Point clicked at ({e.image_x}, {e.image_y})")


def clear_annotations():
    if state["svg_annotations"]:
        state["svg_annotations"].content = ""
        state["annotations"] = {"points": [], "bboxes": [], "masks": []}
        ui.notify("Annotations cleared")


def add_support_image():
    if not state["support_image_path"]:
        ui.notify("Please upload a support image first", color="negative")
        return
    if "annotations" not in state:
        ui.notify("No annotations available to add", color="negative")
        return
    state["support_set"].append(
        {
            "image": state["support_image_path"],
            "annotations": state["annotations"].copy(),
        }
    )
    clear_annotations()
    components["annotation_row"].delete()
    state["support_image_path"] = None
    state["image_widget"] = None
    state["svg_annotations"] = None
    ui.notify("Support image added")


def save_uploaded_file(e, dest_dir: Path) -> Path:
    file_path = dest_dir / e.name
    with open(file_path, "wb") as f:
        f.write(e.content.read())
    return file_path


def show_support_image():
    components["annotation_row"] = ui.element("div").classes("gap-4")
    with components["annotation_row"]:
        ui.label("Annotate your support image").classes("text-lg font-bold")
        state["image_widget"] = ui.interactive_image(
            state["support_image_path"], cross=True
        )
        state["svg_annotations"] = state["image_widget"].add_layer()
        state["polyline"] = None
        state["start_rect"] = None
        state["annotations"] = {"points": {}, "bboxes": {}, "masks": {}}

        state["image_widget"].on_mouse(mouse_handler)


async def load_model(e, model_name=None):
    model_name = model_name or e.value
    components["spinner"] = ui.spinner()
    if not model_name:
        ui.notify("Please select a model", color="negative")
        return
    model_wrapper = await run.cpu_bound(LabelAnything.from_pretrained, model_name)
    state["model"] = model_wrapper.model
    components["spinner"].delete()
    ui.notify(f"Model {model_name} loaded successfully", color="positive")


def handle_support_upload(e):
    file_path = save_uploaded_file(e, Path(UPLOAD_DIR.name))
    state["support_image_path"] = str(file_path)
    ui.notify(f"Support image uploaded: {file_path}")
    components["upload_support"].reset()
    show_support_image()


def handle_query_upload(e):
    file_path = save_uploaded_file(e, Path(UPLOAD_DIR.name))
    state["query_image_path"] = str(file_path)
    ui.notify(f"Query image uploaded: {file_path}")


def run_computation(model, batch):
    with torch.no_grad():
        result = model(batch)
    return result


async def predict():
    print("Predicting...")
    ui.notify("Predicting...", color="info")
    model = state.get("model")
    if not model:
        ui.notify("Please load a model first", color="negative")
        return
    if not state["query_image_path"]:
        ui.notify("Please upload a query image first", color="negative")
        return
    if not state["batch_support"]:
        ui.notify("Please upload support images first", color="negative")
        return
    batch = preprocess_to_batch(
        Image.open(state["query_image_path"]),
        state["batch_support"].copy(),
        size=IMAGE_SIZE,
        device=DEVICE,
    )
    print(f"Batch prepared with {len(state['batch_support'])} support images")
    ui.notify(
        f"Preprocessed query image with {len(state['batch_support'])} support images"
    )
    image_features = await run.cpu_bound(
        get_features, model.image_encoder, batch[BatchKeys.IMAGES]
    )
    print("Extracted image features from the query image")
    ui.notify("Extracted image features from the query image")

    batch[BatchKeys.EMBEDDINGS] = image_features

    state["result"] = await run.cpu_bound(
        run_computation,
        model,
        batch,
    )

    print("Model prediction completed")
    ui.notify("Model prediction completed")

    pred = state["result"][ResultDict.LOGITS].argmax(dim=1)
    plots, titles = plot_seg(
        batch,
        pred,
        COLORS,
        dims=batch[BatchKeys.DIMS],
        classes=state["classes"],
    )

    components["prediction"].clear()
    with components["prediction"]:
        ui.markdown("## 🔮 Prediction Output").classes(
            "text-xl font-semibold text-gray-700 border-b pb-2 mb-4"
        )

        with ui.row().classes("gap-4 w-full"):
            for title, plot in zip(titles, plots):
                with ui.element("div").classes(
                    "flex flex-col flex-1 bg-white p-4 rounded-xl shadow-md items-center"
                ):
                    ui.label(title).classes("text-sm text-gray-600 mb-2")
                    ui.image(Image.fromarray(plot)).classes("rounded w-full").style(
                        "height: auto;"
                    )


def get_features(_model, batch):
    b, n = batch.shape[:2]
    batch = rearrange(batch, "b n c h w -> (b n) c h w")
    with torch.no_grad():
        result = torch.cat(
            [_model(batch[i].unsqueeze(0)) for i in range(batch.shape[0])], dim=0
        )
    result = rearrange(result, "(b n) c h w -> b n c h w", b=b)
    return result


def main():
    ui.page_title("Label Anything")
    ui.markdown("# 🎯 Label Anything Demo")

    # Left Drawer: Model selection
    with ui.left_drawer().classes(
        "bg-gradient-to-b from-blue-100 to-white shadow-md p-4"
    ):
        models = retrieve_models()
        ui.select(
            label="🧠 Select Model",
            options=models,
            on_change=load_model,
            value=models[0],
        ).classes("w-full rounded-md shadow-sm")
        components["spinner"] = ui.element("div")

    # Main layout with two columns
    with ui.element("div").classes("flex flex-col md:flex-row w-full gap-4"):
        # Left Column
        components["left"] = ui.element("div").classes(
            "w-full md:flex-[5] min-w-0 bg-white rounded-xl p-6 shadow-md transition-all duration-300"
        )

        # Right Column
        components["right"] = ui.element("div").classes(
            "w-full md:flex-[5] min-w-0 bg-white rounded-xl p-6 shadow-md transition-all duration-300"
        )

        with components["left"]:
            # Step 1: Upload query image
            ui.markdown("### 📤 Step 1: Upload your query image").classes(
                "font-semibold text-gray-700"
            )
            state["query_image_path"] = ui.upload(
                on_upload=handle_query_upload,
                auto_upload=True,
                label="Upload Query Image",
            ).classes("w-full rounded border border-gray-300")

            # Step 2: Add class labels
            ui.markdown("### 🏷️ Step 2: Add the desired classes").classes(
                "font-semibold text-gray-700"
            )
            label_input = (
                ui.input("Type a class and press Enter")
                .on("keydown.enter", lambda: add_class(label_input, chips))
                .classes("w-full rounded border border-gray-300")
            )
            chips = ui.row().classes("gap-2 flex-wrap mt-2")

            # Step 3: Upload and annotate support images
            ui.markdown("### 🖼️ Step 3: Upload & Annotate Support Images").classes(
                "font-semibold text-gray-700"
            )
            state["prompt"] = (
                ui.radio(["Point", "Rectangle", "Mask"], value="Point")
                .props("inline")
                .classes("mb-2")
            )
            ui.markdown(
                "You can also use *Shift + Click* to draw rectangles, *Ctrl + Click* to draw masks, or just click to add points."
            )

            with ui.element("div").classes("mb-4"):
                components["upload_support"] = ui.upload(
                    on_upload=handle_support_upload,
                    auto_upload=True,
                    label="Upload Support Image",
                ).classes("w-full rounded border border-gray-300")

            with ui.row().classes("gap-2"):
                ui.button("🔚 End Mask", on_click=end_polyline).classes("rounded-lg")
                ui.button("🧹 Clear Annotations", on_click=clear_annotations).classes(
                    "rounded-lg"
                )
                ui.button("➕ Add Support Image", on_click=add_support_image).classes(
                    "rounded-lg"
                )

            ui.separator().classes("my-4 border-t-2")
            ui.button("🚀 Predict", on_click=predict).classes(
                "bg-blue-600 text-white hover:bg-blue-700 font-semibold py-2 px-4 rounded shadow"
            )

    # Populate the right side separately
    with components["right"]:
        # placeholder or functional content for the right panel
        ui.markdown("## 🖼️ Support Set Preview").classes(
            "text-xl font-semibold text-gray-700 border-b pb-2 mb-2"
        )
        components["preview"] = ui.element("div").classes("gap-4")
        components["prediction"] = ui.element("div").classes("gap-4")

    # Initialize models
    app.on_connect(load_model(None, models[0]))

    ui.run()


if __name__ == "__main__":
    main()
