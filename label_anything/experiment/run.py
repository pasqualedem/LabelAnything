import os
from label_anything.data import get_dataloader
from label_anything.logger.text_logger import get_logger
from label_anything.experiment.parameters import parse_params
import sys
import comet_ml
from copy import deepcopy

from label_anything.logger.image_logger import Logger
from label_anything.experiment.train_model import train
from label_anything.models import model_registry

logger = get_logger(__name__)


def comet_experiment(comet_information, args, train_params):
    comet_ml.init(comet_information)
    experiment = comet_ml.Experiment()
    experiment.add_tags(args["tags"])
    experiment.log_parameters(train_params)
    logger = Logger(experiment)
    return logger, experiment


class Run:
    def __init__(self):
        self.kd = None
        self.params = None
        self.dataset = None
        self.experiment = None
        self.comet_logger = None
        self.dataset_params = None
        self.train_params = None
        self.model = None
        if "." not in sys.path:
            sys.path.extend(".")

    def parse_params(self, params):
        self.params = deepcopy(params)

        (
            self.train_params,
            self.dataset_params,
            self.model_params,
        ) = parse_params(self.params)

    def init(self, params: dict):
        self.seg_trainer = None
        self.parse_params(params)
        self.train_params, self.dataset_params, self.model_params = parse_params(params)

        comet_information = {
            "apykey": os.getenv("COMET_API_KEY"),
            "project_name": self.params["experiment"]["name"],
        }

        self.comet_logger, self.experiment = comet_experiment(comet_information, self.params, self.train_params)
        self.url = self.experiment.url
        self.name = self.experiment.name

        self.dataloader = get_dataloader(**self.dataset_params)
        model_name = self.model_params.pop('name')
        self.model = model_registry[model_name](
            **self.model_params
        )
        
    def launch(self):
        train(self.params, self.model, self.dataloader, self.comet_logger, self.experiment, self.train_params)
