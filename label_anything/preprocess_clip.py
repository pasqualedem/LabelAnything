# import open_clip
from argparse import ArgumentParser
import torch
from label_anything.data.coco import LabelAnyThingOnlyImageDataset
from torch.utils.data import DataLoader
import os
import logging
from safetensors.torch import save_file
from ruamel.yaml import YAML
from pathlib import Path
from safetensors.torch import load_file


def load_ruamel(path, typ='safe'):
    yaml = YAML(typ=typ)
    return yaml.load(Path(path))


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--parameters")
    return argparser.parse_args()


def safe_load(path):
    if os.path.exists(path):
        return load_file(path)
    return {}


@torch.no_grad
def extract_and_save_embeddings(
        model,
        dataloader,
        out_dir,
        device='cuda',
):
    os.makedirs(out_dir, exist_ok=True)
    tot_steps = len(dataloader)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )

    model = model.to(device)
    model.eval()
    for ix, batch in enumerate(dataloader):
        img, image_id = batch
        img = img.to(device)
        out = model.encode_image(img).cpu()
        for i in range(out.shape[0]):
            # safe load previous embeddings
            emb_dict = {'clip_embedding': out[i]}
            save_file(
                emb_dict,
                os.path.join(out_dir, f"{image_id[i]}.safetensors"),
            )
        if ix % 10 == 0:
            logging.info(f"Step {ix}/{tot_steps}")


def main(params_path):
    params = load_ruamel(params_path)
    model, _, preprocess = open_clip.create_model_and_transforms(**params['model'])
    dataset = LabelAnyThingOnlyImageDataset(preprocess=preprocess, **params['dataset'])
    dataloader = DataLoader(dataset=dataset, **params['dataloader'])

    extract_and_save_embeddings(
        model=model,
        dataloader=dataloader,
        **params['general'],
    )


