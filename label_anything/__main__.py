from label_anything.models.lam import Lam
from label_anything.parameters import parse_params
from label_anything.utils.utils import load_yaml
from train_model import run
from logger.text_logger import get_logger
from logger.image_logger import Logger
from experiment import comet_experiment
import os
from data.dataset import LabelAnythingDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor


logger = get_logger(__name__)

if __name__ == "__main__":
    args = load_yaml("parameters.yaml")
    logger.info(args)

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
