import torch

from tqdm import tqdm


@torch.no_grad()
def create_image_embeddings(model, dataloader, outfolder, device="cuda"):
    """
    Create image embeddings for all images in dataloader and save them to outfolder.
    """
    for batch in tqdm(dataloader):
        input, gt, additional_information = batch
        img = input['image']
        img = img.to(device)
        out = model(img).cpu()
        for i in range(out.shape[0]):
            torch.save(out[i], outfolder + additional_information[i]['filename'] + '.pth')


def preprocess_images_to_embeddings(encoder_name, args):
    pass
