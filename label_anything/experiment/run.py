import os
import sys
import comet_ml
import tempfile
import subprocess
import uuid
from copy import deepcopy

from label_anything.logger.image_logger import Logger
from label_anything.experiment.train_model import train_and_test
from label_anything.models import model_registry
from label_anything.data import get_dataloaders
from label_anything.logger.text_logger import get_logger

logger = get_logger(__name__)


def parse_params(params_dict):
    train_params = params_dict.get("train_params", {})
    dataset_params = params_dict.get("dataset", {})
    model_params = params_dict.get("model", {})
    dataloader_params = params_dict.get("dataloader", {})

    return train_params, dataset_params, dataloader_params, model_params


def comet_experiment(comet_information: dict, params: dict):
    global logger
    logger_params = deepcopy(params.get("logger", {}))
    logger_params.pop("comet", None)
    if (
        os.environ.get("TMPDIR", None)
        or os.environ.get("TMP", None)
        or os.environ.get("TEMP", None)
    ):
        if os.environ.get("TMPDIR", None):
            tmp_dir = os.environ.get("TMPDIR")
        elif os.environ.get("TMP", None):
            tmp_dir = os.environ.get("TMP")
        else:
            tmp_dir = os.environ.get("TEMP")
        logger.info(
            f"Using {tmp_dir} as temporary directory from environment variables"
        )
        logger_params["tmp_dir"] = tmp_dir
    else:
        tmp_dir = logger_params.get("tmp_dir", None)
        logger.info(
            f"No temporary directory found in environment variables, using {tmp_dir} for images"
        )
    os.makedirs(tmp_dir, exist_ok=True)
    tags = comet_information.pop("tags", [])

    if comet_information.pop("offline"):
        offdir = comet_information.pop("offline_directory", None)
        experiment = comet_ml.OfflineExperiment(
            offline_directory=offdir, **comet_information
        )
    else:
        experiment = comet_ml.Experiment(**comet_information)
    comet_ml.init(comet_information)
    experiment.add_tags(tags)
    experiment.log_parameters(params)

    return Logger(experiment, **logger_params)


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
            self.dataloader_params,
            self.model_params,
        ) = parse_params(self.params)

    def init(self, params: dict):
        self.seg_trainer = None
        self.parse_params(params)
        (
            self.train_params,
            self.dataset_params,
            self.dataloader_params,
            self.model_params,
        ) = parse_params(params)

        comet_params = self.params.get("logger", {}).get("comet", {})
        comet_information = {
            "api_key": os.getenv("COMET_API_KEY"),
            "project_name": self.params["experiment"]["name"],
            **comet_params,
        }

        self.comet_logger = comet_experiment(comet_information, self.params)
        self.url = self.comet_logger.experiment.url
        self.name = self.comet_logger.experiment.name

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            self.dataset_params, self.dataloader_params
        )
        model_name = self.model_params.pop("name")
        self.model = model_registry[model_name](**self.model_params)

    def launch(self):
        train_and_test(
            self.params,
            self.model,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.comet_logger,
            self.train_params,
        )


class ParallelRun(Run):
    slurm_command = "sbatch"
    slurm_script = "launch_run"
    slurm_script_first_parameter = '"--parameters='
    slurm_output = "out/run"
    out_extension = ".out"

    def __init__(self, experiment_uuid: str):
        if "." not in sys.path:
            sys.path.extend(".")
            self.exp_uuid = experiment_uuid

    def launch(self, params: dict):
        tmp_parameters_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_parameters_file.write(str(params).encode())
        tmp_parameters_file.close()
        out_file = f"{self.slurm_output}_{self.exp_uuid}_{str(uuid.uuid4())}_{self.out_extension}"
        subprocess.run(
            [
                self.slurm_command,
                self.slurm_script,
                self.slurm_script_first_parameter + tmp_parameters_file.name + '"',
                out_file,
            ]
        )
