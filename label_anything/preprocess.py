import json
import logging
import os

import numpy as np
import safetensors.torch as safetch
import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import save_file
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from transformers import ViTModel

from label_anything.data import get_mean_std
from label_anything.data.coco import LabelAnyThingOnlyImageDataset
from label_anything.data.transforms import (
    CustomNormalize,
    CustomResize,
    Normalize,
    PromptsProcessor,
    Resize,
)
from label_anything.models import model_registry, build_encoder
from label_anything.utils.utils import ResultDict


def generate_ground_truths(dataset_name, anns_path, outfolder, custom_preprocess=True):
    with open(anns_path, "r") as f:
        anns = json.load(f)
    pp = PromptsProcessor(custom_preprocess=custom_preprocess)
    images = anns["images"]
    annotations = anns["annotations"]

    for image in tqdm(images):
        image_anns = [ann for ann in annotations if ann["image_id"] == image["id"]]
        image_mask = torch.zeros(image["height"], image["width"], dtype=torch.long)
        for ann in image_anns:
            mask = pp.convert_mask(
                ann["segmentation"], image["height"], image["width"]
            ).astype(np.int64)
            mask[mask == 1] = ann["category_id"]
            image_mask = torch.max(image_mask, torch.from_numpy(mask))
        loaded = safetch.load_file(
            os.path.join(outfolder, f"{str(image['id']).zfill(12)}.safetensors")
        )
        loaded[f"{dataset_name}_gt"] = image_mask
        save_file(
            loaded, os.path.join(outfolder, f"{str(image['id']).zfill(12)}.safetensors")
        )


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
    last_block_dir=None,
    device="cuda",
    compile=False,
    custom_preprocess=True,
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
    if last_block_dir is not None:
        os.makedirs(last_block_dir, exist_ok=True)
    print("Model loaded")
    model = model.to(device)
    print("Model moved to device")
    if compile:
        model = torch.compile(model, dynamic=True)
        print("Model compiled")
    preprocess_image = (
        Compose([CustomResize(1024), ToTensor(), CustomNormalize(1024)])
        if custom_preprocess
        else Compose([Resize(1024), ToTensor(), Normalize()])
    )
    dataset = LabelAnyThingOnlyImageDataset(
        directory=directory, preprocess=preprocess_image
    )
    print("Dataset created")
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Dataloader created")

    if last_block_dir is not None:
        create_image_and_neck_embeddings(
            model=model,
            dataloader=dataloader,
            last_hidden_dir=outfolder,
            last_block_dir=last_block_dir,
            device=device,
        )
    else:
        create_image_embeddings(model, dataloader, outfolder, device=device)


@torch.no_grad()
def create_image_and_neck_embeddings(
    model,
    dataloader,
    last_hidden_dir,
    last_block_dir,
    device="cuda",
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )
    n_steps = len(dataloader)

    for idx, batch in enumerate(dataloader):
        img, image_id = batch
        img = img.to(device)
        out = model(img, return_last_block_state=True)
        last_hidden_state = out[ResultDict.LAST_HIDDEN_STATE].cpu()
        last_block_state = out[ResultDict.LAST_BLOCK_STATE].cpu()
        for i in range(last_hidden_state.shape[0]):
            save_file(
                {"embedding": last_hidden_state[i]},
                os.path.join(last_hidden_dir, f"{image_id[i]}.safetensors"),
            )

            save_file(
                {"embedding": last_block_state[i]},
                os.path.join(last_block_dir, f"{image_id[i]}.safetensors"),
            )

        if idx % 10 == 0:
            logging.info(f"Step {idx}/{n_steps}")


@torch.no_grad()
def create_image_embeddings_huggingface(
    model, dataloader, outfolder, device="cuda", image_resolution=480
):
    """
    Create image embeddings for all images in dataloader and save them to outfolder.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )
    n_steps = len(dataloader)
    print("Number of steps needed: ", n_steps)

    for idx, batch in tqdm(enumerate(dataloader), total=n_steps):
        img, image_id = batch
        img = img.to(device)
        out = (
            model(img, interpolate_pos_encoding=True).last_hidden_state[:, 1:, :].cpu()
        )
        out = rearrange(
            out, "b (h w) c -> b c h w", h=image_resolution // model.config.patch_size
        ).contiguous()
        for i in range(out.shape[0]):
            save_file(
                {"embedding": out[i]},
                os.path.join(outfolder, f"{image_id[i]}.safetensors"),
            )


@torch.no_grad
def preprocess_images_to_embeddings_huggingface(
    model_name,
    directory,
    batch_size=1,
    num_workers=0,
    outfolder="data/processed/embeddings",
    device="cuda",
    compile=False,
    image_resolution=480,
    custom_preprocess=True,
    mean_std="default",
):
    os.makedirs(outfolder, exist_ok=True)
    model = ViTModel.from_pretrained(model_name)
    print("Model loaded")
    model = model.to(device)
    print("Model moved to device")
    if compile:
        model = torch.compile(model, dynamic=True)
        print("Model compiled")
    mean, std = get_mean_std(mean_std, mean_std)
    preprocess_image = (
        Compose(
            [
                CustomResize(image_resolution),
                ToTensor(),
                CustomNormalize(image_resolution, mean, std),
            ]
        )
        if custom_preprocess
        else Compose(
            [
                Resize((image_resolution, image_resolution)),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
    )
    dataset = LabelAnyThingOnlyImageDataset(
        directory=directory, preprocess=preprocess_image
    )
    print("Dataset created")
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Dataloader created")
    create_image_embeddings_huggingface(
        model, dataloader, outfolder, device=device, image_resolution=image_resolution
    )


@torch.no_grad()
def preprocess_images_to_feature_pyramids(
    encoder_name,
    directory,
    batch_size=1,
    num_workers=0,
    outfolder="data/processed/embeddings",
    device="cuda",
    compile=False,
    image_resolution=384,
    custom_preprocess=True,
    out_features=["stage2", "stage3", "stage4"],
    mean_std="default",
):
    os.makedirs(outfolder, exist_ok=True)
    encoder = build_encoder.build_encoder(encoder_name)
    print("Model loaded")
    encoder = encoder.to(device)
    print("Model moved to device")
    if compile:
        encoder = torch.compile(encoder, dynamic=True)
        print("Model compiled")
    mean, std = get_mean_std(mean_std, mean_std)
    preprocess_image = (
        Compose(
            [
                CustomResize(image_resolution),
                ToTensor(),
                CustomNormalize(image_resolution, mean, std),
            ]
        )
        if custom_preprocess
        else Compose(
            [
                Resize((image_resolution, image_resolution)),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
    )
    dataset = LabelAnyThingOnlyImageDataset(
        directory=directory, preprocess=preprocess_image
    )
    print("Dataset created")
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Dataloader created")
    for idx, batch in enumerate(tqdm(dataloader)):
        img, image_id = batch
        img = img.to(device)
        out = encoder(img)
        for i in range(img.shape[0]):
            feature_maps = {}
            for j, stage in enumerate(out_features):
                feature_maps[stage] = out.feature_maps[j][i].cpu()
            save_file(
                feature_maps,
                os.path.join(outfolder, f"{image_id[i]}.safetensors"),
            )
        if idx % 10 == 0:
            logging.info(f"Step {idx}/{len(dataloader)}")


def rename_coco20i_json(instances_path: str):
    """Change image filenames of COCO 2014 instances.

    Args:
        instances_path (str): Path to the COCO 2014 instances file.
    """
    with open(instances_path, "r") as f:
        anns = json.load(f)
    for image in anns["images"]:
        image["file_name"] = image["file_name"].split("_")[-1]
    with open(instances_path, "w") as f:
        json.dump(anns, f)
