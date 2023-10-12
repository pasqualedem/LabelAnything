import torch

from tqdm import tqdm
from .data.dataset import LabelAnyThingOnlyImageDataset
from .models import model_registry


@torch.no_grad()
def create_image_embeddings(model, dataloader, outfolder, device="cuda"):
    """
    Create image embeddings for all images in dataloader and save them to outfolder.
    """
    for batch in tqdm(dataloader):
        input, image_id = batch
        img = input['image']
        img = img.to(device)
        out = model(img).cpu()
        for i in range(out.shape[0]):
            torch.save(out[i], outfolder + image_id + '.pth')


def preprocess_images_to_embeddings(
        encoder_name,
        checkpoint,
        use_sam_checkpoint,
        instances_path,
        directory,
        batch_size=1,
        outfolder='data/preprocessed/embeddings'
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
    model = model_registry[encoder_name](checkpoint=checkpoint, use_sam_checkpoint=use_sam_checkpoint)
    dataset = LabelAnyThingOnlyImageDataset(instances_path=instances_path, directory=directory)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    create_image_embeddings(model, dataloader, outfolder)