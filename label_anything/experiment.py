import comet_ml
import os
from label_anything.parameters import parse_params
from label_anything.train_model import train
from label_anything.utils.utils import load_yaml
from logger.image_logger import Logger

from models import model_registry
from data import get_dataloader

# from label_anything.train_model import run
from logger.text_logger import get_logger
from logger.image_logger import Logger


def run_experiment():
    logger = get_logger(__name__)

    args = load_yaml("parameters.yaml")
    # args = load_yaml("parameters.yaml")
    logger.info(args)

    train_params, dataset_params, model_params = parse_params(args)

    comet_information = {
        "apykey": os.getenv("COMET_API_KEY"),
        "project_name": args["experiment"]["name"],
    }
    # hyper_params = {
    #     "batch_size": args["parameters"]["dataset"]["trainloader"]["batch_size"],
    #     # abbiamo un numero di epoche tipo [1, 2, 3] come lo trattiamo?
    #     "num_epochs": args["parameters"]["train_params"]["max_epochs"],
    #     "learning_rate": args["parameters"]["train_params"]["initial_lr"],
    #     "seed": args["parameters"]["train_params"]["seed"],
    #     "loss": args["parameters"]["train_params"]["loss"],
    #     "tags": args["parameters"]["tags"],
    # }

    logger.info("Starting Comet Training")

    comet_logger, experiment = comet_experiment(comet_information, args, train_params)

    dataloader = get_dataloader(**dataset_params)
    model = model_registry[model_params["name"][0]](model_params["checkpoint"][0])

    train(args, model, dataloader, comet_logger, experiment, train_params)


# set configuration
def comet_experiment(comet_information, args, train_params):
    comet_ml.init(comet_information)
    experiment = comet_ml.Experiment()
    experiment.add_tags(args["parameters"]["tags"])
    experiment.log_parameters(train_params)
    logger = Logger(experiment)
    return logger, experiment


if __name__ == "__main__":
    run_experiment()
