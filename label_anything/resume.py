from comet_ml.integration.pytorch import load_model as load
from logger.logger import logger


# Load the model state dict from Comet Registry
def load_model(model, path):
    logger.info(f"Loading Model")
    model.load_state_dict(load(path))


# def load_model_general(model, optimizer, checkpoint):
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     epoch = checkpoint["epoch"]
#     loss = checkpoint["loss"]
#     model.train()
