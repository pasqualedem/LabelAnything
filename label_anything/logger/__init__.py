from logger.clearml_logger import ClearMLLogger
from logger.wandb_logger import WandBSGLogger



LOGGERS = {
    "wandb": WandBSGLogger,
    "clearml": ClearMLLogger,
}
