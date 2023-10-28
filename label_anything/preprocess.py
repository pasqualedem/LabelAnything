import numpy as np
import torch
import os
import torch.nn.functional as F

from tqdm import tqdm
from safetensors.torch import save_file
from torchvision.transforms import Compose, PILToTensor
from label_anything.data.dataset import LabelAnyThingOnlyImageDataset
from label_anything.data.transforms import CustomNormalize, CustomResize
from label_anything.models import model_registry


@torch.no_grad()
def create_image_embeddings(model, dataloader, outfolder, device="cuda"):
    """
    Create image embeddings for all images in dataloader and save them to outfolder.
    """

    for batch in tqdm(dataloader):
        img, image_id = batch
        img = img.to(device)
        out = model(img).cpu()
        for i in range(out.shape[0]):
            save_file(
                {"embedding": out[i]},
                os.path.join(outfolder, f"{image_id[i]}.safetensors"),
            )


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
    preprocess_image = Compose([CustomResize(1024), PILToTensor(), CustomNormalize(1024)])
    dataset = LabelAnyThingOnlyImageDataset(
        directory=directory, preprocess=preprocess_image
    )
    print("Dataset created")
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("Dataloader created")
    create_image_embeddings(model, dataloader, outfolder, device=device)
