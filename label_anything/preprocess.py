import logging
import os

import numpy as np
import torch
import json
import torch.nn.functional as F
from safetensors.torch import save_file
from torchvision.transforms import Compose, PILToTensor
from tqdm import tqdm

from label_anything.data.coco import LabelAnyThingOnlyImageDataset
from label_anything.data.transforms import PromptsProcessor
from label_anything.data.transforms import CustomNormalize, CustomResize
from label_anything.models import model_registry
import safetensors.torch as safetch


def generate_ground_truths(dataset_name, anns_path, outfolder):
    with open(anns_path, "r") as f:
        anns = json.load(f)
    pp = PromptsProcessor()
    images = anns["images"]
    annotations = anns["annotations"]

    for image in tqdm(images):
        image_anns = [ann for ann in annotations if ann["image_id"] == image["id"]]
        image_mask = torch.zeros(image["height"], image["width"], dtype=torch.long)
        for ann in image_anns:
            mask = pp.convert_mask(ann["segmentation"], image["height"], image["width"]).astype(np.int64)
            mask[mask == 1] = ann["category_id"]
            image_mask = torch.max(image_mask, torch.from_numpy(mask))
        loaded = safetch.load_file(
            os.path.join(outfolder, f"{image['id']}.safetensors")
        )
        loaded[f"{dataset_name}_gt"] = image_mask
        save_file(loaded, os.path.join(outfolder, f"{image['id']}.safetensors"))

    
@torch.no_grad()
def create_image_embeddings(model, dataloader, outfolder, device="cuda"):
    """
    Create image embeddings for all images in dataloader and save them to outfolder.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )
    n_steps = len(dataloader)

    for idx, batch in enumerate(dataloader):
        img, image_id = batch
        img = img.to(device)
        out = model(img).cpu()
        for i in range(out.shape[0]):
            save_file(
                {"embedding": out[i]},
                os.path.join(outfolder, f"{image_id[i]}.safetensors"),
            )
        if idx % 10 == 0:
            logging.info(f"Step {idx}/{n_steps}")


def preprocess_images_to_embeddings(
    encoder_name,
    checkpoint,
    use_sam_checkpoint,
    directory,
    batch_size=1,
    num_workers=0,
    outfolder="data/processed/embeddings",
    device="cuda",
    compile=False,
):
    """
    Create image embeddings for all images in dataloader and save them to outfolder.
    Args:
        encoder_name (str): name of the Image Encoder to use
        checkpoint (str): path to the checkpoint
        use_sam_checkpoint (bool): tells if the checkpoint is a SAM checkpoint
        instances_path (str): path to the instances file
        directory (str): directory of the images
        batch_size (int): batch size for the dataloader
        num_workers (int): number of workers for the dataloader
        outfolder (str): folder to save the embeddings
    """
    os.makedirs(outfolder, exist_ok=True)
    model = model_registry[encoder_name](
        checkpoint=checkpoint, use_sam_checkpoint=use_sam_checkpoint
    )
    print("Model loaded")
    model = model.to(device)
    print("Model moved to device")
    if compile:
        model = torch.compile(model, dynamic=True)
        print("Model compiled")
    preprocess_image = Compose(
        [CustomResize(1024), PILToTensor(), CustomNormalize(1024)]
    )
    dataset = LabelAnyThingOnlyImageDataset(
        directory=directory, preprocess=preprocess_image
    )
    print("Dataset created")
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Dataloader created")
    create_image_embeddings(model, dataloader, outfolder, device=device)
