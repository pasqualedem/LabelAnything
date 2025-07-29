from tempfile import TemporaryDirectory
from label_anything.demo.builtin_ng import built_in_dataset
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
from label_anything.demo.utils import COLORS, get_features, retrieve_models, run_computation, sanitize
from label_anything.demo.preprocess_ng import preprocess_support_set
from label_anything.models.build_lam import LabelAnything
from label_anything.utils.utils import ResultDict

IMAGE_SIZE = 1024
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
RGB_TO_TEXT_COLORS = {
    # rbg to hex
    f"rgba({c[0]}, {c[1]}, {c[2]}, 1)": c
    for c in COLORS[1:]
}
TEXT_COLORS = list(RGB_TO_TEXT_COLORS.keys())


def preview_support_set(batch):
    # Clear previous preview if it exists
    if "preview" in app.storage.tab["components"]:
        app.storage.tab["components"]["preview"].clear()

    with app.storage.tab["components"]["preview"]:
        # Scrollable image row
        with ui.row().classes("gap-4 overflow-x-auto whitespace-nowrap pb-2"):
            for i, _ in enumerate(app.storage.tab["support_set"]):
                img = batch[BatchKeys.IMAGES][0][i]
                masks = batch[BatchKeys.PROMPT_MASKS][0][i]
                bboxes = batch[BatchKeys.PROMPT_BBOXES][0][i]
                points = batch[BatchKeys.PROMPT_POINTS][0][i]

                img = get_image(img)
                img = draw_all(
                    img, masks=masks, boxes=bboxes, points=points, colors=[COLORS[0]] + [RGB_TO_TEXT_COLORS[c] for c in app.storage.tab["taken_colors"].values()]
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
    classes = app.storage.tab["classes"].copy()
    classes = sanitize(classes)
    support_set = sanitize(app.storage.tab["support_set"])
    image_size = app.storage.tab.get("image_size", IMAGE_SIZE)
    
    app.storage.tab["batch_support"] = preprocess_support_set(
        support_set,
        classes,
        size=image_size,
        custom_preprocess=True,
        device="cpu",
    )
    for k, v in app.storage.tab["batch_support"].items():
        print(f"{k}: instance of {type(v)}")
    preview_support_set(app.storage.tab["batch_support"])


UPLOAD_DIR = TemporaryDirectory(prefix="label_anything_uploads", delete=False)


def set_current_class(value):
    if value in app.storage.tab["classes"]:
        app.storage.tab["current_class"] = value
        ui.notify(f"Current class set to: {value}")
    else:
        ui.notify(f'Class "{value}" not found in available classes', color="negative")


def add_class(label_input: ui.input, chips):
    color = app.storage.tab["available_colors"].pop()
    app.storage.tab["taken_colors"][label_input.value] = color
    app.storage.tab["current_class"] = label_input.value
    if not app.storage.tab["available_colors"]:
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
    app.storage.tab["classes"].append(label_input.value)
    label_input.value = ""


def remove_class(chip: ui.chip, chips):
    chips.remove(chip)
    ui.notify(f'Removed class: {chip.text} from {app.storage.tab["classes"]}')
    app.storage.tab["classes"].remove(chip.text)
    app.storage.tab["available_colors"].add(app.storage.tab["taken_colors"].pop(chip.text))


def end_polyline():
    current_class = app.storage.tab.get("current_class")
    if not current_class:
        ui.notify("Please select a class first by clicking on it", color="negative")
        return
    if not app.storage.tab["polyline"]:
        ui.notify("No polyline started", color="negative")
        return

    last_line = f"<line x1=\"{app.storage.tab['polyline'][-2]}\" y1=\"{app.storage.tab['polyline'][-1]}\" x2=\"{app.storage.tab['polyline'][0]}\" y2=\"{app.storage.tab['polyline'][1]}\" stroke=\"{app.storage.tab['taken_colors'][current_class]}\" stroke-width=\"3\" />"
    app.storage.tab["svg_annotations"].content += last_line

    if current_class not in app.storage.tab["annotations"]["masks"]:
        app.storage.tab["annotations"]["masks"][current_class] = []
    app.storage.tab["annotations"]["masks"][current_class].append(
        app.storage.tab["polyline"]
    )
    app.storage.tab["polyline"] = None


def mouse_handler(e: events.MouseEventArguments):
    selected_prompt = app.storage.tab["prompt"]
    current_class = app.storage.tab.get("current_class")
    color = app.storage.tab["taken_colors"].get(current_class)
    if not color:
        ui.notify("Please select a class first by clicking on it", color="negative")
        return
    # ui.notify(
    #     f"State prompt: {state['prompt'].value}, Mouse event: {e.type}, Coordinates: ({e.image_x}, {e.image_y}, Class: {current_class}, Color: {color})"
    # )
    if e.shift or selected_prompt == "Rectangle":  # Add a rectangle
        if app.storage.tab["start_rect"]:
            # Finish rectangle and draw it
            x1, y1 = app.storage.tab["start_rect"]
            x2, y2 = e.image_x, e.image_y
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            rect_svg = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="3" />'
            app.storage.tab["svg_annotations"].content += rect_svg
            if current_class not in app.storage.tab["annotations"]["bboxes"]:
                app.storage.tab["annotations"]["bboxes"][current_class] = []
            app.storage.tab["annotations"]["bboxes"][current_class].append(
                (x, y, x + w, y + h)
            )
            app.storage.tab["start_rect"] = None
        else:
            app.storage.tab["start_rect"] = (e.image_x, e.image_y)
    elif e.ctrl or selected_prompt == "Mask":  # Add a polyline
        if app.storage.tab["polyline"]:
            app.storage.tab[
                "svg_annotations"
            ].content += f'<line x1="{app.storage.tab["polyline"][-2]}" y1="{app.storage.tab["polyline"][-1]}" x2="{e.image_x}" y2="{e.image_y}" stroke="{color}" stroke-width="3" />'
            app.storage.tab["polyline"].extend([e.image_x, e.image_y])
            # Add the line to annotations
            # Notify the user
            ui.notify(f"Line drawn to ({e.image_x}, {e.image_y})")
        else:
            app.storage.tab["polyline"] = [e.image_x, e.image_y]
    else:
        app.storage.tab[
            "svg_annotations"
        ].content += f'<circle cx="{e.image_x}" cy="{e.image_y}" r="15" fill="none" stroke="{color}" stroke-width="4" />'
        if current_class not in app.storage.tab["annotations"]["points"]:
            app.storage.tab["annotations"]["points"][current_class] = []
        app.storage.tab["annotations"]["points"][current_class].append(
            (e.image_x, e.image_y)
        )


def clear_annotations():
    if app.storage.tab["svg_annotations"]:
        app.storage.tab["svg_annotations"].content = ""
        app.storage.tab["annotations"] = {"points": [], "bboxes": [], "masks": []}
        ui.notify("Annotations cleared")


def add_support_image():
    if not app.storage.tab["support_image_path"]:
        ui.notify("Please upload a support image first", color="negative")
        return
    if "annotations" not in app.storage.tab:
        ui.notify("No annotations available to add", color="negative")
        return
    app.storage.tab["support_set"].append(
        {
            "image": app.storage.tab["support_image_path"],
            "annotations": app.storage.tab["annotations"].copy(),
        }
    )
    clear_annotations()
    app.storage.tab["components"]["annotation_row"].delete()
    app.storage.tab["support_image_path"] = None
    app.storage.tab["image_widget"] = None
    app.storage.tab["svg_annotations"] = None
    ui.notify("Support image added")


def save_uploaded_file(e, dest_dir: Path) -> Path:
    file_path = dest_dir / e.name
    with open(file_path, "wb") as f:
        f.write(e.content.read())
    return file_path


def show_support_image():
    app.storage.tab["components"]["annotation_row"] = ui.element("div").classes("gap-4")
    with app.storage.tab["components"]["annotation_row"]:
        ui.label("Annotate your support image").classes("text-lg font-bold")
        app.storage.tab["image_widget"] = ui.interactive_image(
            app.storage.tab["support_image_path"], cross=True
        )
        app.storage.tab["svg_annotations"] = app.storage.tab["image_widget"].add_layer()
        app.storage.tab["polyline"] = None
        app.storage.tab["start_rect"] = None
        app.storage.tab["annotations"] = {"points": {}, "bboxes": {}, "masks": {}}

        app.storage.tab["image_widget"].on_mouse(mouse_handler)


async def load_model(e, model_name=None):
    model_name = model_name or e.value
    app.storage.tab["components"]["spinner"] = ui.spinner()
    if not model_name:
        ui.notify("Please select a model", color="negative")
        return
    model_wrapper = await run.cpu_bound(LabelAnything.from_pretrained, model_name)
    app.storage.tab["model"] = model_wrapper.model
    app.storage.tab["model_config"] = model_wrapper.config
    app.storage.tab["components"]["spinner"].delete()
    ui.notify(f"Model {model_name} loaded successfully", color="positive")

def set_device(device):
    ui.notify("Bugged, not implemented yet, setting device to cpu", color="negative")
    return
    app.storage.tab["device"] = device
    if device == "cuda":
        if not torch.cuda.is_available():
            ui.notify("CUDA is not available, switching to CPU", color="negative")
            app.storage.tab["device"] = "cpu"
    else:
        app.storage.tab["device"] = "cpu"
    print(f"Device set to: {app.storage.tab['device']}")


def handle_support_upload(e):
    file_path = save_uploaded_file(e, Path(UPLOAD_DIR.name))
    app.storage.tab["support_image_path"] = str(file_path)
    ui.notify(f"Support image uploaded: {file_path}")
    app.storage.tab["components"]["upload_support"].reset()
    show_support_image()


def handle_query_upload(e):
    file_path = save_uploaded_file(e, Path(UPLOAD_DIR.name))
    app.storage.tab["query_image_path"] = str(file_path)
    ui.notify(f"Query image uploaded: {file_path}")


async def predict():
    components = app.storage.tab["components"]
    print("Predicting...")
    ui.notify("Predicting...", color="info")
    model = app.storage.tab.get("model")
    if not model:
        ui.notify("Please load a model first", color="negative")
        return
    if not app.storage.tab["query_image_path"]:
        ui.notify("Please upload a query image first", color="negative")
        return
    if not app.storage.tab["batch_support"]:
        ui.notify("Please upload support images first", color="negative")
        return
    batch = preprocess_to_batch(
        Image.open(app.storage.tab["query_image_path"]),
        app.storage.tab["batch_support"].copy(),
        size=IMAGE_SIZE,
        device="cpu",
    )
    # Make batch obserable-free by turning observables into regular lists
    classes = batch[BatchKeys.CLASSES]
    classes = [[v for v in c] for c in classes]
    batch[BatchKeys.CLASSES] = classes

    print(f"Batch prepared")
    ui.notify(f"Batch prepared")
    # image_features = await run.cpu_bound(
    #     get_features, model.image_encoder, batch[BatchKeys.IMAGES], app.storage.tab["device"]
    # )
    # print("Extracted image features from the query image")
    # ui.notify("Extracted image features from the query image")

    # batch[BatchKeys.EMBEDDINGS] = image_features

    app.storage.tab["result"] = await run.cpu_bound(
        run_computation,
        model,
        batch,
        app.storage.tab["device"],
    )

    print("Model prediction completed")
    ui.notify("Model prediction completed")

    pred = app.storage.tab["result"][ResultDict.LOGITS].argmax(dim=1)
    plots, titles = plot_seg(
        batch,
        pred,
        COLORS,
        dims=batch[BatchKeys.DIMS],
        classes=app.storage.tab["classes"],
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


async def page_init():
    print("Page initialized")
    app.storage.tab.update(
        {
            "query_image_path": None,
            "support_image_path": None,
            "image_widget": None,
            "classes": [],
            "current_class": None,
            "support_set": observables.ObservableList([], on_change=set_batch_support),
            "batch_support": None,
            "prompt": "Point",
            "svg_annotations": None,
            "polyline": None,
            "start_rect": None,
            "annotations": None,
            "model": None,
            "result": None,
            "components": {"builtin": {}},
            "device": DEVICE,
            "available_colors": set(TEXT_COLORS.copy()),
            "taken_colors": {}
        }
    )


@ui.page('/')
async def index():
    await ui.context.client.connected()
    await page_init() 
    components = app.storage.tab["components"]

    ui.page_title("Label Anything")
    ui.markdown("# 🏷️ Label Anything")

    # Model list and selection
    models = retrieve_models()
    app.storage.tab['model'] = models[0]

    def on_model_change(e):
        app.storage.tab['model'] = e.value
        load_model(e)

    with ui.left_drawer().classes("bg-gradient-to-b from-blue-100 to-white shadow-md p-4"):
        ui.select(
            label="🧠 Select Model",
            options=models,
            on_change=on_model_change,
            value=models[0],
        ).classes("w-full rounded-md shadow-sm")

        components["spinner"] = ui.element("div")
        ui.switch(
            "Use GPU",
            value=app.storage.tab["device"] == "cuda",
            on_change=lambda x: set_device("cuda" if x.value else "cpu"),
        )

    # Tabs: "Segment your images" and "Segment from COCO"
    with ui.tabs().classes("w-full") as tabs:
        tab1 = ui.tab("🖼️ Segment your images")
        tab2 = ui.tab("📦 Segment from COCO")

    with ui.tab_panels(tabs, value=tab1).classes("w-full"):
        with ui.tab_panel(tab1):
            with ui.element("div").classes("flex flex-col md:flex-row w-full gap-4"):
                components["left"] = ui.element("div").classes(
                    "w-full md:flex-[5] min-w-0 bg-white rounded-xl p-6 shadow-md transition-all duration-300"
                )
                components["right"] = ui.element("div").classes(
                    "w-full md:flex-[5] min-w-0 bg-white rounded-xl p-6 shadow-md transition-all duration-300"
                )

                with components["left"]:
                    ui.markdown("### 📤 Step 1: Upload your query image").classes("font-semibold text-gray-700")
                    ui.upload(
                        on_upload=handle_query_upload,
                        auto_upload=True,
                        label="Upload Query Image",
                    ).classes("w-full rounded border border-gray-300")

                    ui.markdown("### 🎯 Step 2: Add the desired classes").classes("font-semibold text-gray-700")
                    label_input = ui.input("Type a class and press Enter").on(
                        "keydown.enter", lambda: add_class(label_input, chips)
                    ).classes("w-full rounded border border-gray-300 px-3")
                    chips = ui.row().classes("gap-2 flex-wrap mt-2")

                    ui.markdown("### 🖼️ Step 3: Upload & Annotate Support Images").classes("font-semibold text-gray-700")
                    ui.radio(["Point", "Rectangle", "Mask"], value="Point").props("inline").classes("mb-2").bind_value(app.storage.tab, "prompt")
                    ui.markdown("You can also use *Shift + Click* to draw rectangles, *Ctrl + Click* to draw masks, or just click to add points.")

                    with ui.element("div").classes("mb-4"):
                        components["upload_support"] = ui.upload(
                            on_upload=handle_support_upload,
                            auto_upload=True,
                            label="Upload Support Image",
                        ).classes("w-full rounded border border-gray-300")

                    with ui.row().classes("gap-2"):
                        ui.button("🔚 End Mask", on_click=end_polyline).classes("rounded-lg")
                        ui.button("🧹 Clear Annotations", on_click=clear_annotations).classes("rounded-lg")
                        ui.button("➕ Add Support Image", on_click=add_support_image).classes("rounded-lg")

                    ui.separator().classes("my-4 border-t-2")
                    ui.button("🚀 Predict", on_click=predict).classes(
                        "bg-blue-600 text-white hover:bg-blue-700 font-semibold py-2 px-4 rounded shadow"
                    )

                with components["right"]:
                    ui.markdown("## 🖼️ Support Set Preview").classes(
                        "text-xl font-semibold text-gray-700 border-b pb-2 mb-2"
                    )
                    components["preview"] = ui.element("div").classes("gap-4")
                    components["prediction"] = ui.element("div").classes("gap-4")

        with ui.tab_panel(tab2):
            # Pass currently selected model to built-in dataset UI
            built_in_dataset()

    # Load initial model once
    await load_model(None, models[0])


def main():
    print("✅ Starting NiceGUI app...")
    ui.run(favicon="🏷️", port=8501, reload=False)
    
    
if __name__ in {"__main__", "__mp_main__"}:
    main()

