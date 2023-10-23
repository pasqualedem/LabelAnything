import numpy as np
import torch
import os
import torch.nn.functional as F

from tqdm import tqdm
from safetensors.torch import save_file
from torchvision.transforms.functional import resize, to_pil_image
from data.dataset import LabelAnyThingOnlyImageDataset
from models import model_registry


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def preprocess_tensor(x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        x = (x - pixel_mean) / pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

def preprocess_image(img):
        """
        Preprocess the image before getting fed to the Image Encoder
        """
        img = img.convert("RGB")
        long_size = 1024
        img = np.array(img)

        target_size = get_preprocess_shape(img.shape[0], img.shape[1], long_size)

        input_image = np.array(resize(to_pil_image(img), target_size))

        input_image_torch = torch.tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        return preprocess_tensor(input_image_torch)


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
            save_file({"embedding": out[i]}, os.path.join(outfolder, image_id[i] + '.safetensors'))


def preprocess_images_to_embeddings(
        encoder_name,
        checkpoint,
        use_sam_checkpoint,
        directory,
        batch_size=1,
        outfolder='data/processed/embeddings',
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
        outfolder (str): folder to save the embeddings
    """
    os.makedirs(outfolder, exist_ok=True)
    model = model_registry[encoder_name](checkpoint=checkpoint, use_sam_checkpoint=use_sam_checkpoint)
    print("Model loaded")
    model = model.to(device)
    print("Model moved to device")
    if compile:
        model = torch.compile(model, dynamic=True)
        print("Model compiled") 
    dataset = LabelAnyThingOnlyImageDataset(directory=directory, preprocess=preprocess_image)
    print("Dataset created")
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    print("Dataloader created")
    create_image_embeddings(model, dataloader, outfolder, device=device)