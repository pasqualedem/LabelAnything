from label_anything.logger.text_logger import get_logger
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from label_anything.parameters import parse_params
import sys
from copy import deepcopy
import traceback

from label_anything.utils.utils import dict_to_yaml_string

logger = get_logger(__name__)


class Run:
    def __init__(self):
        self.kd = None
        self.params = None
        self.dataset = None
        self.train_callbacks = None
        self.val_callbacks = None
        self.test_callbacks = None
        self.dataset_params = None
        self.seg_trainer = None
        self.train_params = None
        self.test_params = None
        self.run_params = None
        self.phases = None
        self.carbon_tracker = None
        if "." not in sys.path:
            sys.path.extend(".")

    def parse_params(self, params):
        self.params = deepcopy(params)
        self.phases = params["phases"]

        (
            self.train_params,
            self.test_params,
            self.dataset_params,
            callbacks,
            self.kd,
        ) = parse_params(self.params)
        self.train_callbacks, self.val_callbacks, self.test_callbacks = callbacks
        self.run_params = params.get("run_params") or {}

    def _init_carbon_tracker(self):
        try:
            self.carbon_tracker = EmissionsTracker(
                output_dir=self.seg_trainer.sg_logger._local_dir, log_level="warning"
            )
        except ConnectionError:
            logger.warning(
                "CodeCarbon is not connected to a server, using offline tracker"
            )
            self.carbon_tracker = OfflineEmissionsTracker(
                output_dir=self.seg_trainer.sg_logger._local_dir,
                log_level="warning",
                country_iso_code="ITA",
            )
        self.carbon_tracker.start()

    def init(self, params: dict):
        self.seg_trainer = None
        try:
            self.parse_params(params)
            # trainer_class = KDSegTrainer if kd else SegmentationTrainer
            trainer_class = KDEzTrainer if self.kd else EzTrainer
            self.seg_trainer = trainer_class(
                project_name=self.params["experiment"]["name"],
                group_name=self.params["experiment"]["group"],
                ckpt_root_dir=self.params["experiment"]["tracking_dir"]
                or "experiments",
            )
            self.dataset = self.seg_trainer.init_dataset(
                params["dataset_interface"],
                dataset_params=deepcopy(self.dataset_params),
            )
            self.seg_trainer.init_model(params, False, None)
            self.seg_trainer.init_loggers(
                {"in_params": params}, deepcopy(self.train_params)
            )
            logger.info(f"Input params: \n\n {dict_to_yaml_string(params)}")
            if params.get("print_model_summary", True):
                self.seg_trainer.print_model_summary()
            self._init_carbon_tracker()
        except Exception as e:
            if self.seg_trainer is not None and self.seg_trainer.sg_logger is not None:
                self.seg_trainer.sg_logger.close(really=True, failed=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e
