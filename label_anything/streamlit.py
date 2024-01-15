import streamlit as st
import torch
import numpy as np
import io
from PIL import Image


@st.cache_resource
def load_model(model_file, file_type):
    if file_type == "pth":
        buffer = io.BytesIO(model_file.getbuffer())
        return torch.load(buffer)
    elif file_type == "npz":
        buffer = io.BytesIO(model_file.getbuffer())
        return np.load(buffer)


def main():
    st.title("Label Anything")
    st.sidebar.title("Settings")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg"])
    row = st.columns(2)

    # load model
    uploaded_file = st.sidebar.file_uploader("Choose a model file", type=["pth", "npz"])
    if uploaded_file is not None:
        model = load_model(uploaded_file, uploaded_file.name.split(".")[-1])
        st.sidebar.write("Model loaded successfully!")

    # load image
    image_placeholder = row[0].empty()
    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        image_placeholder.image(image, caption="Uploaded Image", use_column_width=True)

    # show prediction
    if uploaded_file is not None and uploaded_image is not None:
        button_row = st.columns([1, 1, 1])
        if button_row[1].button("Predict"):
            prediction = model(image)
            row[1].write("Prediction")
            row[1].image(prediction, caption="Prediction Image", use_column_width=True)


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
