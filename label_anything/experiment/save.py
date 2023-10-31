from comet_ml.integration.pytorch import log_model

from label_anything.logger.text_logger import get_logger


logger = get_logger(__name__)


def save_model(experiment, model, model_name):
    logger.info(f"Saving Model {model_name}")
    log_model(experiment, model, model_name)
    logger.info(f"Finished Saving Model")


def save_model_for_resume(epoch, model, optimizer, loss, experiment, model_name):
    logger.info(f"Saving Model for Resume {epoch}")
    model_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    log_model(experiment, model_checkpoint, model_name)
