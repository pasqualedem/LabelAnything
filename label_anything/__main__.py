# from label_anything.parameters import parse_args
from preprocess import preprocess_images_to_embeddings

from logger.text_logger import get_logger
from logger.image_logger import Logger
from experiment import comet_experiment
import os

from data.dataset import LabelAnythingDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from models.lam import Lam
from parameters import parse_params
from utils.utils import load_yaml


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("operation", help="Select the operation to perform (preprocess, train, test)")
parser.add_argument("--encoder", default="vit_h", help="Select the encoder to use")
parser.add_argument("--checkpoint", default="vit_h.pth", help="Select the file to use as checkpoint")
parser.add_argument("--use_sam_checkpoint", action="store_true", help="Select if the checkpoint is a SAM checkpoint")
parser.add_argument("--compile", action="store_true", help="Select if the model should be compiled")
parser.add_argument("--directory", default="data/raw/train2017", help="Select the file to use as checkpoint")
parser.add_argument("--batch_size", default=1, help="Batch size for the dataloader")
parser.add_argument("--outfolder", default="data/processed/embeddings", help="Folder to save the embeddings")


logger = get_logger(__name__)

if __name__ == "__main__":
  
    args = load_yaml("parameters.yaml")
    logger.info(args)
    args = parser.parse_args()

    if args.operation == "preprocess":
        preprocess_images_to_embeddings(
            encoder_name=args.encoder,
            checkpoint=args.checkpoint,
            use_sam_checkpoint=args.use_sam_checkpoint,
            directory=args.directory,
            batch_size=args.batch_size,
            outfolder=args.outfolder,
            compile=args.compile,
        )
        exit()

    comet_information = {
        "apykey": os.getenv("COMET_API_KEY"),
        "project_name": args["experiment"]["name"],
    }

    hyper_params = {
        "batch_size": args["parameters"]["dataset"]["trainloader"]["batch_size"],
        # abbiamo un numero di epoche tipo [1, 2, 3] come lo trattiamo?
        "num_epochs": args["parameters"]["train_params"]["max_epochs"],
        "learning_rate": args["parameters"]["train_params"]["initial_lr"],
        "seed": args["parameters"]["train_params"]["seed"],
        "loss": args["parameters"]["train_params"]["loss"],
        "tags": args["parameters"]["tags"],
    }

    print(hyper_params)
    logger.info("Starting Comet Training")

    comet_logger, experiment = comet_experiment(comet_information, hyper_params)

    preprocess = Compose(
        [
            ToTensor(),
            # Resize((1000, 1000)),
        ]
    )

    dataset = LabelAnythingDataset(
        instances_path="label_anything/data/lvis_v1_train.json",
        preprocess=preprocess,
        num_max_examples=10,
        j_index_value=0.1,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn
    )
    model = Lam()

    run(args, model, dataloader, comet_logger, experiment, hyper_params)
