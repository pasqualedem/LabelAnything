from label_anything.data.utils import BatchKeys, PromptType
from label_anything.demo.utils import get_features
from label_anything.demo.utils import COLORS, get_data, run_computation
from label_anything.demo.visualize import draw_all, get_image, plot_seg
from label_anything.experiment.substitution import Substitutor
from label_anything.utils.utils import ResultDict
from nicegui import ui, app, run
from PIL import Image


async def predict():
    components = app.storage.tab["components"]["builtin"]
    print("Predicting...")
    ui.notify("Predicting...", color="info")
    model = app.storage.tab.get("model")
    if not model:
        ui.notify("Please load a model first", color="negative")
        return
    if not app.storage.tab["batch"]:
        ui.notify("Please load the images first", color="negative")
        return
    batch, gt = app.storage.tab["batch"]
    substitutor = Substitutor(substitute=False)
    substitutor.reset(batch=(batch, gt))
    batch, gt = next(iter(substitutor))
    
    # Make batch obserable-free by turning observables into regular lists
    classes = batch.pop(BatchKeys.CLASSES, None)
    batch.pop(BatchKeys.INTENDED_CLASSES, None)
    batch.pop(BatchKeys.IMAGE_IDS, None)
    
    # image_features = await run.cpu_bound(
    #     get_features, model.image_encoder, batch[BatchKeys.IMAGES]
    # )
    # print("Extracted image features from the query image")
    # ui.notify("Extracted image features from the query image")

    # batch[BatchKeys.EMBEDDINGS] = image_features
    
    # for key in batch:
    #     print(f"Batch key: {key}, Type: {type(batch[key])}")
    
    # import pickle
    # for key, value in batch.items():
    #     try:
    #         pickle.dumps(value)  # Check if the value can be pickled
    #         pickle.dumps(key)  # Check if the key can be pickled
    #         print(f"Key: {key} is picklable")
    #     except Exception as e:
    #         print(f"Error pickling {key}: {e} - Type: {type(value)}")
    
    # pickle.dumps(model)
    # print("Model pickled successfully")
    
    # pickle.dumps(device)
    # print("Device pickled successfully")
    
    # pickle.dumps(run_computation)
    # print("run_computation pickled successfully")

    result = await run.cpu_bound(
        run_computation,
        model,
        batch,
        app.storage.tab["device"],
    )

    print("Model prediction completed")
    ui.notify("Model prediction completed")

    pred = result[ResultDict.LOGITS].argmax(dim=1)
    plots, titles = plot_seg(
        batch,
        pred,
        COLORS,
        dims=batch[BatchKeys.DIMS],
        classes=classes[0][0],
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
                    

def load_image():
    print("Loading images...")
    dataloader = app.storage.tab["dataset"]
    if not dataloader:
        ui.notify("Dataset not loaded", color="red")
        return
    
    batch, _ = next(dataloader, None)
    print("Batch loaded")
    app.storage.tab["batch"] = batch
    batch, _ = batch
    
    components = app.storage.tab["components"]["builtin"]
    # Clear previous preview if it exists
    if "preview" in components:
        components["preview"].clear()

    with components["preview"]:
        # Scrollable image row
        with ui.row().classes("gap-4 overflow-x-auto whitespace-nowrap pb-2"):
            for i, _ in enumerate(batch[BatchKeys.IMAGES][0]):
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
                    if i == 0:
                        ui.label("Query Image").classes(
                            "text-sm text-gray-600 mt-2 font-semibold"
                        )
                    else:
                        ui.label(f"Support Image {i}").classes(
                            "text-sm text-gray-600 mt-2"
                        )


def built_in_dataset():
    components = app.storage.tab["components"]["builtin"]
    with ui.element("div").classes("flex flex-col md:flex-row w-full gap-4"):
        components["left"] = ui.element("div").classes(
            "w-full md:flex-[5] min-w-0 bg-white rounded-xl p-6 shadow-md transition-all duration-300"
        )
        components["right"] = ui.element("div").classes(
            "w-full md:flex-[5] min-w-0 bg-white rounded-xl p-6 shadow-md transition-all duration-300"
        )
        
    with components["left"]:
        ui.markdown("## 📦 Dataset Parameters").classes(
                        "text-xl font-semibold text-gray-700 border-b pb-2 mb-2"
                    )

        with ui.row().classes("w-full flex-wrap gap-4"):
            n_ways_input = ui.number("Number of ways", value=1, min=1, max=20).props('outlined').classes("w-full md:w-1/2")
            n_shots_input = ui.number("Number of shots", value=1, min=1, max=20).props('outlined').classes("w-full md:w-1/2")
            use_examples_checkbox = ui.checkbox("Use shots as examples", value=False).classes("w-full md:w-1/2")

            point_checkbox = ui.checkbox("Points", value=False).classes("w-full md:w-1/2")
            bbox_checkbox = ui.checkbox("Bounding Box", value=False).classes("w-full md:w-1/2")
            mask_checkbox = ui.checkbox("Mask", value=True).classes("w-full md:w-1/2")

        with ui.row().classes("w-full flex-wrap gap-4"):
            max_points_input = ui.number("Max points per annotation", value=5, min=1, max=20).props('outlined').classes("w-full md:w-1/2")
            fold_input = ui.number("Fold", value=0, min=0, max=3).props('outlined').classes("w-full md:w-1/2")

        with ui.row().classes("w-full flex-wrap gap-4"):
            all_categories_checkbox = ui.checkbox("Use all categories in examples", value=False).classes("w-full md:w-1/2")
            class_sampling_checkbox = ui.checkbox("Class based sampling", value=True).classes("w-full md:w-1/2")

    async def load_dataset():
        if 'model_config' not in app.storage.tab:
            ui.notify("Please load a model first", color="red")
            return
        
        n_ways = n_ways_input.value
        n_shots = n_shots_input.value
        n_examples = n_shots if use_examples_checkbox.value else None
        shots = None if n_examples else n_shots
        image_size = app.storage.tab['model_config']["image_size"]
        preprocess = app.storage.tab['model_config']["custom_preprocess"]
        all_cats = all_categories_checkbox.value
        max_points = max_points_input.value
        fold = fold_input.value
        prompt_types = [
            prompt for prompt in [
                PromptType.POINT if point_checkbox.value else None,
                PromptType.BBOX if bbox_checkbox.value else None,
                PromptType.MASK if mask_checkbox.value else None
            ] if prompt is not None
        ]

        dataloader = await get_data(n_ways, shots, n_examples, image_size, preprocess,
                              all_cats, prompt_types, max_points, fold, class_based_sampling=class_sampling_checkbox.value)
        if not dataloader:
            return

        ui.notify("COCO Annotations loaded", color='green')

        app.storage.tab['dataset'] = iter(dataloader)
        app.storage.tab['datasets'] = dataloader.dataset.datasets
        
    with components["left"]:
        ui.separator().classes("my-4 border-t-2")
        with ui.row().classes("gap-2"):
            ui.button('📤 Load dataset', on_click=load_dataset).classes(
                            "bg-blue-600 text-white hover:bg-blue-700 font-semibold py-2 px-4 rounded shadow"
                        )
            ui.button('🐈 Load images', on_click=load_image).classes(
                            "bg-blue-600 text-white hover:bg-blue-700 font-semibold py-2 px-4 rounded shadow"
                        )
            ui.button('🔮 Predict', on_click=predict).classes(
                            "bg-blue-600 text-white hover:bg-blue-700 font-semibold py-2 px-4 rounded shadow"
                        )
            
    with components["right"]:
        ui.markdown("## 🖼️ Episode Preview").classes(
            "text-xl font-semibold text-gray-700 border-b pb-2 mb-2"
        )
        components["preview"] = ui.element("div").classes("gap-4")
        components["prediction"] = ui.element("div").classes("gap-4")