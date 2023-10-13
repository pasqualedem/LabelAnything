# from label_anything.parameters import parse_args
from .preprocess import preprocess_images_to_embeddings

from train_model import run
from logger.text_logger import get_logger
from logger.image_logger import Logger
from experiment import comet_experiment
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("operation", help="Select the operation to perform (preprocess, train, test)")
parser.add_argument("--encoder", default="vit_h", help="Select the encoder to use")
parser.add_argument("--checkpoint", default="vit_h", help="Select the file to use as checkpoint")
parser.add_argument("--use_sam_checkpoint", action="store_true", help="Select the file to use as checkpoint")
parser.add_argument("--instances_path", default="instances.json", help="Select the file to use as checkpoint")
parser.add_argument("--directory", default="data/raw/train2017", help="Select the file to use as checkpoint")
parser.add_argument("--batch_size", default=1, help="Batch size for the dataloader")
parser.add_argument("--outfolder", default="data/preprocessed/embeddings", help="Folder to save the embeddings")


logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting Comet Training")
    args = parser.parse_args()

    if args.operation == "preprocess":
        preprocess_images_to_embeddings(
            encoder_name=args.encoder,
            checkpoint=args.checkpoint,
            use_sam_checkpoint=args.use_sam_checkpoint,
            instances_path=args.instances_path,
            directory=args.directory,
            batch_size=args.batch_size,
            outfolder=args.outfolder
        )

    comet_information = {
        "apykey": os.getenv("COMET_API_KEY"),
        "project_name": args.name,
    }

    hyper_params = {
        "batch_size": args.dataset["trainloader"]["batch_size"]["value"],
        # abbiamo un numero di epoche tipo [1, 2, 3] come lo trattiamo?
        "num_epochs": args.train_params["max_epochs"]["value"],
        "learning_rate": args.train_params["learning_rate"]["value"],
        "seed": args.train_params["seed"]["value"],
        "loss": args.train_params["loss"]["value"],
        "tags": args.train_params["model"]["tags"],
    }

    comet_logger, experiment = comet_experiment(comet_information, hyper_params)
    # TODO: dataset e dataloader da importare
    dataset = None
    dataloader = None
    model = None

    run(args, model, dataloader, comet_logger, experiment, hyper_params)
