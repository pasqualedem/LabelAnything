from label_anything.parameters import parse_args
from train_model import run
from logger.logger import logger
from experiment import comet_experiment
import os


if __name__ == "__main__":
    logger.info("Starting Comet Training")
    args = parse_args()

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
