import comet_ml
import os
from label_anything.parameters import parse_params
from label_anything.train_model import train
from label_anything.utils.utils import load_yaml
from label_anything.models import model_registry
from label_anything.data import get_dataloader
from label_anything.logger.text_logger import get_logger
from label_anything.logger.image_logger import Logger

logger = get_logger(__name__)


def run_experiment(checkpoint, use_sam_checkpoint):
    logger.info("Running experiment")
    args = load_yaml("parameters.yaml")
    # args = load_yaml("parameters.yaml")
    train_params, dataset_params, model_params = parse_params(args)

    comet_information = {
        "apykey": os.getenv("COMET_API_KEY"),
        "project_name": args["experiment"]["name"],
    }

    comet_logger, experiment = comet_experiment(comet_information, args, train_params)

    dataloader = get_dataloader(**dataset_params)
    model = model_registry[model_params["name"][0]](
        checkpoint=checkpoint,
        use_sam_checkpoint=use_sam_checkpoint,
    )
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
    run_experiment(
        checkpoint="/home/emanuele/Workspace/dottorato/LabelAnything/sam_vit_h.pth",
        use_sam_checkpoint=True,
    )
